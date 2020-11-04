import hydra
import numpy as np
from omegaconf import DictConfig
from typing import Tuple, List, Any, Dict
import golem as glm
import torch
from torch import multiprocessing as mp
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
import time
import itertools as it
import os
from time import sleep
from .net import NetHackNet as Net
from .env import create_env, ResettingEnvironment
from .vtrace import from_logits, from_importance_weights

import tracemalloc


def nested_map(f, n):
    if isinstance(n, tuple) or isinstance(n, list):
        return n.__class__(nested_map(f, sn) for sn in n)
    elif isinstance(n, dict):
        return {k: nested_map(f, v) for k, v in n.items()}
    else:
        return f(n)


def compute_baseline_loss(advantages):
    return 0.5 * torch.sum(advantages ** 2)


def compute_entropy_loss(logits):
    """Return the entropy loss, i.e., the negative entropy of the policy."""
    policy = F.softmax(logits, dim=-1)
    log_policy = F.log_softmax(logits, dim=-1)
    return torch.sum(policy * log_policy)


def compute_policy_gradient_loss(logits, actions, advantages):
    cross_entropy = F.nll_loss(
        F.log_softmax(torch.flatten(logits, 0, 1), dim=-1),
        target=torch.flatten(actions, 0, 1),
        reduction="none",
    )
    cross_entropy = cross_entropy.view_as(advantages)
    return torch.sum(cross_entropy * advantages.detach())


def _create_buffers(unroll_length, observation_space, num_actions, initial_state):
    # Get specimens to infer shapes and dtypes.
    size = (unroll_length + 1,)
    samples = {k: torch.from_numpy(v) for k, v in observation_space.sample().items()}

    specs = {
        key: dict(size=size + sample.shape, dtype=sample.dtype)
        for key, sample in samples.items()
    }
    specs.update(
        reward=dict(size=size, dtype=torch.float32),
        done=dict(size=size, dtype=torch.bool),
        episode_return=dict(size=size, dtype=torch.float32),
        episode_step=dict(size=size, dtype=torch.int32),
        policy_logits=dict(size=size + (num_actions,), dtype=torch.float32),
        baseline=dict(size=size, dtype=torch.float32),
        last_action=dict(size=size, dtype=torch.int64),
        action=dict(size=size, dtype=torch.int64),
    )
    buffers = {key: None for key in specs}
    for key in buffers:
        buffers[key] = torch.empty(**specs[key])
    state = initial_state 
    buffers["initial_agent_state"] = state

    return buffers


def learner(
    init: Tuple[
        glm.distributed.World,
        glm.distributed.RpcGroup,
        glm.buffers.DistributedBuffer,
        glm.servers.PushPullModelServer,
    ],
    cfg: DictConfig,
):
    # TODO: Load the batch while you backward pass
    # Could use Torch dataset
    writer = SummaryWriter()
    hparams_dict = glm.utils.flatten(cfg)
    writer.add_hparams(hparams_dict, {})
    logger = glm.utils.default_logger
    torch.set_num_threads(8)
    logger.info("Booting an Impala learner")
    world, impala_group, replay_buffer, server = init
    is_leader = world.rank == 0
    logger.info("My friends: " + str(world.get_members()))
    logger.info("Env: " + str(cfg.env.task))
    env = create_env(cfg.env.task, savedir=None)
    observation_space = env.observation_space
    action_space = env.action_space
    model = Net(observation_space, action_space.n, cfg.model.lstm)
    del env
    optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=cfg.training.learning_rate,
        momentum=cfg.training.momentum,
        eps=cfg.training.epsilon,
        alpha=cfg.training.alpha,
    )
    B, T = cfg.training.batch_size, cfg.training.unroll_length
    total_steps = cfg.training.total_steps
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    model = model.to(device)
    moving_episode_returns = glm.utils.Moving(size=100)

    def lr_lambda(epoch):
        # This should also be multiplied by the amount of learner
        return 1 - min(epoch * T * B, total_steps) / total_steps

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    impala_group.barrier(is_leader)
    prof = glm.utils.SimpleProfiler()
    buffer_keys = None
    with prof(category="sleeping"):
        while replay_buffer.all_size() < B:
            logger.info(
                f"The size of the replay buffer ({replay_buffer.all_size()}) is smaller then {B}. Waiting..."
            )
            sleep(5)

    for iteration in it.count():
        with prof(category="rpc-sample"):
            bsize, buffers = replay_buffer.sample_batch(B, 'actor', 8)
            if bsize == 0:
                logger.warning("Batch size is 0!")
                sleep(1)
                continue
        with prof(category="preprocessing-1"):
            if buffer_keys is None:
                buffer_keys = buffers[0].keys()
                buffer_keys = list(filter(lambda k: k != "initial_agent_state", buffer_keys))
            batch = {
                key: torch.stack([buffers[m][key] for m in range(bsize)], dim=1)
                for key in buffer_keys
            }
        with prof(category="preprocessing-2"):
            initial_agent_state = (
                torch.cat(ts, dim=1)
                for ts in zip(
                    *[buffers[m]["initial_agent_state"] for m in range(bsize)]
                )
            )
        with prof(category="preprocessing-3"):
            batch = {
                k: t.to(device=device, non_blocking=True) for k, t in batch.items()
            }
        with prof(category="preprocessing-4"):
            initial_agent_state = tuple(
                t.to(device=device, non_blocking=True) for t in initial_agent_state
            )
        with prof(category="forward_pass"):
            learner_outputs, unused_state = model(batch, initial_agent_state)

        with prof(category="backward_pass"):
            # Take final value function slice for bootstrapping.
            bootstrap_value = learner_outputs["baseline"][-1]

            # Move from obs[t] -> action[t] to action[t] -> obs[t].
            batch = {key: tensor[1:] for key, tensor in batch.items()}
            learner_outputs = {
                key: tensor[:-1] for key, tensor in learner_outputs.items()
            }

            rewards = batch["reward"]
            clipped_rewards = torch.tanh(rewards)

            discounts = (~batch["done"]).float() * cfg.training.discounting

            vtrace_returns = from_logits(
                behavior_policy_logits=batch["policy_logits"],
                target_policy_logits=learner_outputs["policy_logits"],
                actions=batch["action"],
                discounts=discounts,
                rewards=clipped_rewards,
                values=learner_outputs["baseline"],
                bootstrap_value=bootstrap_value,
            )

            pg_loss = compute_policy_gradient_loss(
                learner_outputs["policy_logits"],
                batch["action"],
                vtrace_returns.pg_advantages,
            )
            baseline_loss = cfg.training.baseline_cost * compute_baseline_loss(
                vtrace_returns.vs - learner_outputs["baseline"]
            )
            entropy_loss = cfg.training.entropy_cost * compute_entropy_loss(
                learner_outputs["policy_logits"]
            )

            total_loss = pg_loss + baseline_loss + entropy_loss

            episode_returns = batch["episode_return"][batch["done"]]
            ep_returns = tuple(episode_returns.cpu().numpy())
            moving_episode_returns.addm(ep_returns)
            mean_ep_returns_moving, var_ep_returns_moving, mean_abs_ep_returns = moving_episode_returns.stats(add_abs=True) 
            stats = {
                "real_batch_size": bsize,
                "episode_returns": ep_returns,
                "mean_episode_return": mean_ep_returns_moving,
                "mean_abs_episode_return": mean_abs_ep_returns,
                "var_episode_return": var_ep_returns_moving,
                "ep_in_mini_batch": len(ep_returns),
                "total_loss": total_loss.item(),
                "pg_loss": pg_loss.item(),
                "baseline_loss": baseline_loss.item(),
                "entropy_loss": entropy_loss.item(),
            }
            logger.info(stats)

            writer.add_scalar("Training/TotalLoss", total_loss.item())
            writer.add_scalar("Training/PGLoss", pg_loss.item())
            writer.add_scalar("Training/BaselineLoss", baseline_loss.item())
            writer.add_scalar("Training/EntropyLoss", entropy_loss.item())
            writer.add_scalar("Training/MeanEpisodeReturns", mean_ep_returns_moving)
            writer.add_scalar("Training/MeanAbsEpisodeReturns", mean_abs_ep_returns)
            writer.add_scalar("Training/VarEpisodeReturns", var_ep_returns_moving)
            writer.add_scalar("Training/TotalEpisodes", moving_episode_returns.total)

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), cfg.training.grad_norm_clipping
            )
            optimizer.step()
            scheduler.step()
            if iteration % 20 == 0:
                impala_group.registered_sync("increment_parameter_updates")
                impala_group.registered_sync("increment_step_processed")
                if (
                    world.rank == 0
                    and impala_group.registered_sync("get_step_processed")
                    > cfg.training.total_steps
                ):
                    impala_group.registered_sync("turn_global_switch_off")
                    break
                elif impala_group.registered_sync("get_global_switch") is False:
                    break
                global_timer = impala_group.registered_sync("get_global_timer")
                samples_collected = impala_group.registered_sync("get_samples_collected")
                parameter_updates = impala_group.registered_sync("get_parameter_updates")
                steps_processed = impala_group.registered_sync("get_step_processed")
                steps_sampled_s = ( 
                                samples_collected
                                / global_timer
                )
                param_updates_s = ( 
                                parameter_updates
                                / global_timer
                )
                steps_processed_s = ( 
                                steps_processed
                                / global_timer
                )
                logger.info(
                        "{:.2f} steps sampled/s".format(
                            steps_sampled_s
                            )
                        )
                logger.info(
                        "{:.2f} param updates/s".format(
                            param_updates_s
                            )
                        )
                logger.info(
                        "{:.2f} steps processed/s".format(
                            steps_processed_s
                            )
                        )

                logger.info(
                        "total steps sampled = {:.2f}".format(
                            samples_collected
                            )
                        )
                logger.info(
                        "total param updates = {:.2f}".format(
                            parameter_updates
                            )
                        )
                logger.info(
                        "total steps processed = {:.2f}".format(
                            steps_processed
                            )
                        )
                logger.info(prof)
                writer.add_scalar("Speed/StepSampledS", steps_sampled_s)
                writer.add_scalar("Speed/ParamUpdatesS", param_updates_s)
                writer.add_scalar("Speed/StepProcessedS", steps_processed_s)

                writer.add_scalar("Progress/StepSampled", samples_collected)
                writer.add_scalar("Progress/ParamUpdates", parameter_updates)
                writer.add_scalar("Progress/StepProcessed", steps_processed)

        with prof(category="rpc-parameter-server"):
            if iteration % 20 == 0:
                server.push(model)

    logger.info(prof)
    logger.info(
        "{:.2f} steps sampled/s".format(
            impala_group.registered_sync("get_samples_collected")
            / impala_group.registered_sync("get_global_timer")
        )
    )
    logger.info(
        "{:.2f} param updates/s".format(
            impala_group.registered_sync("get_parameter_updates")
            / impala_group.registered_sync("get_global_timer")
        )
    )
    logger.info(
        "{:.2f} steps processed/s".format(
            impala_group.registered_sync("get_step_processed")
            / impala_group.registered_sync("get_global_timer")
        )
    )
    # Have a barrier at the end to make sure everything is sync before exiting
    impala_group.barrier(is_leader)


def actor(
    init: Tuple[
        glm.distributed.World,
        glm.distributed.RpcGroup,
        glm.buffers.DistributedBuffer,
        glm.servers.PushPullModelServer,
    ],
    cfg: DictConfig,
):
    torch.set_num_threads(4)
    logger = glm.utils.default_logger
    logger.info("Booting an Impala actor")
    world, impala_group, replay_buffer, server = init
    is_leader = world.rank == 0
    logger.info("My friends: " + str(world.get_members()))
    logger.info(f"Saving NLE data in {os.getcwd()}")
    env = create_env(cfg.env.task, savedir=os.getcwd())
    observation_space = env.observation_space
    action_space = env.action_space
    env = ResettingEnvironment(env)
    model = Net(observation_space, action_space.n, cfg.model.lstm)
    env_output = env.initial()
    agent_state = model.initial_state(batch_size=1)
    agent_output, unused_state = model(env_output, agent_state)
    impala_group.barrier(is_leader)
    prof = glm.utils.SimpleProfiler()
    server.pull(model)
    buffers = _create_buffers(
        cfg.training.unroll_length, observation_space, action_space.n, agent_state
    )
    for iteration in it.count():
        # Reset the buffers to prevent the memory leak
        if iteration % 100 == 0:
            with prof(category="recreating_buffers"):
                del buffers
                buffers = _create_buffers(
                    cfg.training.unroll_length, observation_space, action_space.n, agent_state
                )
        # Write old rollout end.
        with prof(category="buffer"):
            for key in env_output:
                buffers[key][0, ...] = env_output[key]
            for key in agent_output:
                buffers[key][0, ...] = agent_output[key]
            for i, tensor in enumerate(agent_state):
                buffers["initial_agent_state"][i][...] = tensor

        # Do new rollout.
        for t in range(cfg.training.unroll_length):
            with prof(category="model"):
                with torch.no_grad():
                    agent_output, agent_state = model(env_output, agent_state)

            with prof(category="env"):
                # Reseed when episode restarts
                env_output = env.step(agent_output["action"], force_seed=np.random.randint(0,1000000))

            with prof(category="buffer"):
                for key in env_output:
                    buffers[key][t + 1, ...] = env_output[key]
                for key in agent_output:
                    buffers[key][t + 1, ...] = agent_output[key]

        with prof(category="replay-buffer"):
            replay_buffer.append(buffers)
        with prof(category="rpc-increment"):
            if iteration % 20 == 0:
                impala_group.registered_sync("increment_sample_collected")
        with prof(category="rpc-pulling"):
            if iteration % 20 == 0:
                logger.info("Pulling model....")
                logger.info(prof)
                server.pull(model)
                logger.info("Done!")

        with prof(category="rpc-switch"):
            if iteration % 20 == 0:
                if impala_group.registered_sync("get_global_switch") is False:
                    break

    logger.info(prof)
    # Have a barrier at the end to make sure everything is synchronized before exiting
    impala_group.barrier(is_leader)


def init(cfg: DictConfig):
    logger = glm.utils.default_logger
    world = glm.distributed.create_world_with_env()
    is_leader = world.rank == 0
    impala_group = world.create_rpc_group("impala", world.get_members(), lead=is_leader)
    logger.info("RPC group setup")
    replay_buffer = glm.buffers.DistributedBuffer(
        buffer_name="buffer", group=impala_group,
        buffer_size=20
    )
    logger.info("Buffer setup")
    server = glm.servers.model_server_helper(model_num=1, lead=is_leader)[0]
    logger.info("Model Server setup")
    # Counters and Switch
    if is_leader:
        logger.info("I am the rank 0. Creating counters")

        sample_counter = glm.widgets.Counter(step=cfg.training.unroll_length * 20)
        impala_group.register("increment_sample_collected", sample_counter.count)
        impala_group.register("get_samples_collected", sample_counter.get)

        parameter_update_counter = glm.widgets.Counter(step=20)
        impala_group.register(
            "increment_parameter_updates", parameter_update_counter.count
        )
        impala_group.register("get_parameter_updates", parameter_update_counter.get)

        step_processed_counter = glm.widgets.Counter(
            step=cfg.training.batch_size * cfg.training.unroll_length * 20
        )
        impala_group.register("increment_step_processed", step_processed_counter.count)
        impala_group.register("get_step_processed", step_processed_counter.get)

        global_switch = glm.widgets.Switch(state=True)
        impala_group.register("turn_global_switch_off", global_switch.off)
        impala_group.register("get_global_switch", global_switch.get)

        global_timer = glm.widgets.Timer()
        impala_group.register("get_global_timer", global_timer.end)

    return (world, impala_group, replay_buffer, server)


hydra.main()(glm.launcher.launch(init, {"actor": actor, "learner": learner}))()
