import hydra
import traceback
import math
import numpy as np
from omegaconf import DictConfig
from typing import Tuple, List, Any, Dict
import golem as glm
import torch
from torch import multiprocessing as mp
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import multiprocessing as mp
import time
import itertools as it
import os
import threading
from time import sleep
from .net import GridNet as Net
from .net import GridStateEmbeddingNet as EmbeddingNet
from .net import MinigridForwardDynamicsNet  as ForwardNet
from .net import MinigridInverseDynamicsNet  as InverseNet
from .env import create_env, ResettingEnvironment
from .vtrace import from_logits, from_importance_weights

import tracemalloc

torch.manual_seed(0)
np.random.seed(0)
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


def compute_forward_dynamics_loss(pred_next_emb, next_emb):
    forward_dynamics_loss = torch.norm(pred_next_emb - next_emb, dim=2, p=2)
    return torch.sum(torch.mean(forward_dynamics_loss, dim=1))


def compute_inverse_dynamics_loss(pred_actions, true_actions):
    inverse_dynamics_loss = F.nll_loss(
        F.log_softmax(torch.flatten(pred_actions, 0, 1), dim=-1), 
        target=torch.flatten(true_actions, 0, 1), 
        reduction='none')
    inverse_dynamics_loss = inverse_dynamics_loss.view_as(true_actions)
    return torch.sum(torch.mean(inverse_dynamics_loss, dim=1))

def _create_buffers(unroll_length, observation_space, num_actions, initial_state):
    # Get specimens to infer shapes and dtypes.
    size = (unroll_length + 1,)
    specs = dict(
        frame=dict(size=size + observation_space.shape, dtype=torch.float),
        reward=dict(size=size, dtype=torch.float32),
        done=dict(size=size, dtype=torch.bool),
        episode_return=dict(size=size, dtype=torch.float32),
        episode_step=dict(size=size, dtype=torch.int32),
        policy_logits=dict(size=size + (num_actions,), dtype=torch.float32),
        baseline=dict(size=size, dtype=torch.float32),
        last_action=dict(size=size, dtype=torch.int64),
        action=dict(size=size, dtype=torch.int64),
        episode_state_count=dict(size=size, dtype=torch.float32)
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
    world, impala_group, replay_buffer, server = init
    is_leader = world.rank == 0
    if is_leader:
        writer = SummaryWriter('summary')
        hparams_dict = glm.utils.flatten(cfg)
        writer.add_hparams(hparams_dict, {})
    logger = glm.utils.default_logger
    torch.set_num_threads(8)
    logger.info("Booting an Impala learner")
    logger.info("My friends: " + str(world.get_members()))

    logger.info("Validating the config...")
    if cfg.training.checkpointing_frequency % 20 != 0:
        logger.error("cfg.training.checkpointing_frequency needs to be a multiple of 20. Aborting.")
        raise Exception("Invalid Config")

    logger.info("Init the default process group for DDP")
    process_group_host = os.environ["MASTER_ADDR"]
    process_group_port = int(os.environ["MASTER_PORT"]) + 1
    init_method = f"tcp://{process_group_host}:{process_group_port}"
    logger.info("Init method: " + init_method)
    torch.distributed.init_process_group("gloo", init_method=init_method, world_size=cfg.run.tasks.learner.processes, rank=world.rank)

    logger.info("Env: " + str(cfg.env.task))
    env = create_env(cfg.env.task)
    observation_space = env.observation_space
    action_space = env.action_space
    model = Net(observation_space, action_space.n, cfg.model.embedding_size, cfg.model.lstm)
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
    model.share_memory()
    # Create the model initial_state before wrapping it with DDD
    initial_states = model.initial_state(batch_size=B)
    # Devices Ids should be replaced by local ranks

    # Initialize the embedding model
    embedding_net = EmbeddingNet(observation_space)
    embedding_net = embedding_net.to(device)
    embedding_net.share_memory()
    embedding_optimizer = torch.optim.RMSprop(
        embedding_net.parameters(),
        lr=cfg.training.learning_rate,
        momentum=cfg.training.momentum,
        eps=cfg.training.epsilon,
        alpha=cfg.training.alpha,
    )


    # Initialize the dynamics models
    forward_net = ForwardNet(action_space.n)
    forward_net = forward_net.to(device)
    forward_net.share_memory()
    forward_optimizer = torch.optim.RMSprop(
        forward_net.parameters(),
        lr=cfg.training.learning_rate,
        momentum=cfg.training.momentum,
        eps=cfg.training.epsilon,
        alpha=cfg.training.alpha,
    )

    inverse_net = InverseNet(action_space.n)
    inverse_net = inverse_net.to(device)
    inverse_net.share_memory()
    inverse_optimizer = torch.optim.RMSprop(
        inverse_net.parameters(),
        lr=cfg.training.learning_rate,
        momentum=cfg.training.momentum,
        eps=cfg.training.epsilon,
        alpha=cfg.training.alpha,
    )

    def lr_lambda(epoch):
        # This should also be multiplied by the amount of learner
        return 1 - min(epoch * T * B, total_steps) / total_steps

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    embedding_scheduler = torch.optim.lr_scheduler.LambdaLR(embedding_optimizer, lr_lambda)
    forward_scheduler = torch.optim.lr_scheduler.LambdaLR(forward_optimizer, lr_lambda)
    inverse_scheduler = torch.optim.lr_scheduler.LambdaLR(inverse_optimizer, lr_lambda)
    # Register the evaluator callback
    current_iteration = 0
    if is_leader:
        def log_evaluator_results(results):
            global_step = current_iteration
            writer.add_scalar("Eval/EvalMeanReturns", results["mean_returns"], global_step=global_step)
            writer.add_scalar("Eval/EvalVarReturns", results["var_returns"], global_step=global_step)
            writer.add_scalar("Eval/EvalMeanSteps", results["mean_steps"], global_step=global_step)
            writer.add_scalar("Eval/EvalVarSteps", results["var_steps"], global_step=global_step)
        impala_group.register("log_evaluator_results", log_evaluator_results)
        def save_evaluator_model(model_state_dict, mean_returns):
            global_step = current_iteration
            logger.info("Saving best evaluator model checkpoint...")
            writer.add_scalar("Eval/BestMeanReturns", mean_returns, global_step=global_step)
            torch.save(dict(
                model=model_state_dict,
                ), 'best_eval.pt')
        impala_group.register("save_evaluator_model", save_evaluator_model)
    # What iteration are we on?

    # Restoring logic
    if int(os.environ.get("GOLEM_RESTORE", False)):
        experiment_id = os.environ["GOLEM_EXPERIMENT_ID"]
        logger.info("Restoring!")
        logger.info(f"Experiment ID: {experiment_id}")
        try:
            checkpoint = torch.load("checkpoint.pt")
        except Exception:
            logger.error("Could not restore checkpoint!")
            raise Exception("Could not restore checkpoint")
        if is_leader:
            optimizer.load_state_dict(checkpoint['optimizer'])
            model.load_state_dict(checkpoint['model'])
            scheduler.load_state_dict(checkpoint['scheduler'])

            embedding_optimizer.load_state_dict(checkpoint['embedding_optimizer'])
            embedding_net.load_state_dict(checkpoint['embedding_net'])
            embedding_scheduler.load_state_dict(checkpoint['embedding_scheduler'])

            forward_optimizer.load_state_dict(checkpoint['forward_optimizer'])
            forward_net.load_state_dict(checkpoint['forward_net'])
            forward_scheduler.load_state_dict(checkpoint['forward_scheduler'])

            inverse_optimizer.load_state_dict(checkpoint['inverse_optimizer'])
            inverse_net.load_state_dict(checkpoint['inverse_net'])
            inverse_scheduler.load_state_dict(checkpoint['inverse_scheduler'])

            impala_group.registered_sync('set_samples_collected', args=(checkpoint['samples_collected'],))
            impala_group.registered_sync('set_step_processed', args=(checkpoint['steps_processed'],))
            impala_group.registered_sync('set_parameter_updates', args=(checkpoint['parameter_updates'],))
            current_iteration = checkpoint['iteration']
        else:
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            embedding_optimizer.load_state_dict(checkpoint['embedding_optimizer'])
            embedding_scheduler.load_state_dict(checkpoint['embedding_scheduler'])
            forward_optimizer.load_state_dict(checkpoint['forward_optimizer'])
            forward_scheduler.load_state_dict(checkpoint['forward_scheduler'])
            inverse_optimizer.load_state_dict(checkpoint['inverse_optimizer'])
            inverse_scheduler.load_state_dict(checkpoint['inverse_scheduler'])
            current_iteration = checkpoint['iteration']
        logger.info("Restore successful!")
        logger.info(f"Current iteration: {current_iteration}")

    model = DDP(model, device_ids=[0])
    embedding_net = DDP(embedding_net, device_ids=[0])
    forward_net = DDP(forward_net, device_ids=[0])
    inverse_net = DDP(inverse_net, device_ids=[0])

    logger.info("Waiting on the barrier")
    impala_group.barrier(is_leader)
    logger.info("All the processes joined!")
    while replay_buffer.all_size() < B:
        logger.info(
            f"The size of the replay buffer ({replay_buffer.all_size()}) is smaller then {B}. Waiting..."
        )
        sleep(5)
    # Main thread does rpc sampling, stats, and logging. Pushes to a preprocess queue. Pulls from a stats queue.
    # Preprocess thread pulls from a filled buffer index queue, it publishes to a done buffer index queue 
    # Learner thread pulls from the done buffer index queue, learns, and publishes to a learned buffer index queue (with learning stats)
    # Main thread pulls from the learner queue, fills the corresponding buffer, and publishes that stats if needed

    # Create buffer from example
    _, b = replay_buffer.sample_batch(B, 'actor', 8)
    sample_buffer = b[0]
    buffer_size = (cfg.training.unroll_length, B)
    buffer_keys = sample_buffer.keys()
    buffer_keys = list(filter(lambda k: k != "initial_agent_state", buffer_keys))
    specs = {
        key: dict(size=buffer_size + sample_buffer[key].shape, dtype=sample_buffer[key].dtype)
        for key in buffer_keys
    }
    buffers = {key: [] for key in specs}
    for _ in range(cfg.training.num_buffers):
        for key in buffers:
            buffers[key].append(torch.empty(**specs[key]).share_memory_())

    initial_agent_state_buffers = []
    for _ in range(cfg.training.num_buffers):
        state = initial_states
        for t in state:
            t.share_memory_()
        initial_agent_state_buffers.append(state)

    del initial_states

    # Non local variables

    stats = {}

    # Queues
    ctx = mp.get_context("fork")
    full_queue = ctx.SimpleQueue()
    learned_queue = ctx.SimpleQueue()

    def learn():
        nonlocal stats
        moving_episode_returns = glm.utils.Moving(size=100)
        try:
            while True: 
                buffer_index = full_queue.get()
                batch = {key: buffers[key][buffer_index] for key in buffer_keys}
                initial_agent_state = initial_agent_state_buffers[buffer_index]
                batch = {
                    k: t.to(device=device, non_blocking=True) for k, t in batch.items()
                }
                initial_agent_state = tuple(
                    t.to(device=device, non_blocking=True) for t in initial_agent_state
                )
                # Intrinsic reward
                count_rewards = batch["episode_state_count"][1:]
                state_emb = embedding_net(batch['frame'][:-1])
                next_state_emb = embedding_net(batch['frame'][1:])
                pred_next_state_emb = forward_net(state_emb, batch['action'][1:])
                pred_action = inverse_net(state_emb, next_state_emb)
                control_rewards = torch.norm(next_state_emb - state_emb, dim=2, p=2)
                intrinsic_rewards = control_rewards * count_rewards
                intrinsic_rewards *= cfg.training.intrinsic_reward_coef
                mean_intrinsic_reward = torch.mean(intrinsic_rewards)
                var_intrinsic_reward = torch.var(intrinsic_rewards)
                # Losses
                forward_dynamics_loss = cfg.training.forward_loss_coef * compute_forward_dynamics_loss(pred_next_state_emb, next_state_emb)
                inverse_dynamics_loss = cfg.training.inverse_loss_coef * compute_inverse_dynamics_loss(pred_action, batch['action'][1:])

                learner_outputs, unused_state = model(batch, initial_agent_state)
                # Take final value function slice for bootstrapping.
                bootstrap_value = learner_outputs["baseline"][-1]
                # Move from obs[t] -> action[t] to action[t] -> obs[t].
                batch = {key: tensor[1:] for key, tensor in batch.items()}
                learner_outputs = {
                    key: tensor[:-1] for key, tensor in learner_outputs.items()
                }
                rewards = batch["reward"]
                rewards = rewards + intrinsic_rewards
                clipped_rewards = rewards.clamp(-1, 1)
                # clipped_rewards = rewards
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
                total_loss = pg_loss + baseline_loss + entropy_loss + forward_dynamics_loss + inverse_dynamics_loss
                episode_returns = batch["episode_return"][batch["done"]]
                ep_returns = tuple(episode_returns.cpu().numpy())
                moving_episode_returns.addm(ep_returns)
                mean_ep_returns_moving, var_ep_returns_moving, mean_abs_ep_returns = moving_episode_returns.stats(add_abs=True) 
                optimizer.zero_grad()
                embedding_optimizer.zero_grad()
                forward_optimizer.zero_grad()
                inverse_optimizer.zero_grad()
                total_loss.backward()
                grad_norm_model = sum(p.grad.data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
                grad_norm_embedding = sum(p.grad.data.norm(2).item() ** 2 for p in embedding_net.parameters() if p.grad is not None) ** 0.5
                grad_norm_forward = sum(p.grad.data.norm(2).item() ** 2 for p in forward_net.parameters() if p.grad is not None) ** 0.5
                grad_norm_inverse = sum(p.grad.data.norm(2).item() ** 2 for p in inverse_net.parameters() if p.grad is not None) ** 0.5
                stats = {
                    "real_batch_size": bsize,
                    #"episode_returns": ep_returns,
                    "mean_ep_returns_moving": mean_ep_returns_moving,
                    "mean_abs_ep_return": mean_abs_ep_returns,
                    "grad_norm_model": grad_norm_model,
                    "grad_norm_embedding": grad_norm_embedding,
                    "grad_norm_forward": grad_norm_forward,
                    "grad_norm_inverse": grad_norm_inverse,
                    "var_ep_returns_moving": var_ep_returns_moving,
                    "ep_in_mini_batch": len(ep_returns),
                    "mean_intrinsic_reward": mean_intrinsic_reward.item(),
                    "var_intrinsic_reward": var_intrinsic_reward.item(),
                    "total_loss": total_loss.item(),
                    "pg_loss": pg_loss.item(),
                    "baseline_loss": baseline_loss.item(),
                    "entropy_loss": entropy_loss.item(),
                    "forward_loss": forward_dynamics_loss.item(),
                    "inverse_loss": inverse_dynamics_loss.item(),
                    "total_ep": moving_episode_returns.total
                }
                learned_queue.put(buffer_index)
                nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.training.grad_norm_clipping
                )
                nn.utils.clip_grad_norm_(
                    embedding_net.parameters(), cfg.training.grad_norm_clipping
                )
                nn.utils.clip_grad_norm_(
                    forward_net.parameters(), cfg.training.grad_norm_clipping
                )
                nn.utils.clip_grad_norm_(
                    inverse_net.parameters(), cfg.training.grad_norm_clipping
                )
                optimizer.step()
                embedding_optimizer.step()
                forward_optimizer.step()
                inverse_optimizer.step()
                scheduler.step()
                embedding_scheduler.step()
                forward_scheduler.step()
                inverse_scheduler.step()
        except Exception as e:
            logger.error("Error in learner thread: " + str(e))
            logger.error(traceback.print_exc())
            learned_queue.put(-1)

    # Save once
    if is_leader:
        samples_collected = impala_group.registered_sync("get_samples_collected")
        parameter_updates = impala_group.registered_sync("get_parameter_updates")
        steps_processed = impala_group.registered_sync("get_step_processed")
        # Checkpoint
        # - Model params
        # - Optimizer state dict
        # - steps_processed
        # - param_parameter_updates
        # - samples_collected
        logger.info("Saving initial checkpoint...")
        torch.save(dict(
            model=model.module.state_dict(),
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            embedding_net=embedding_net.module.state_dict(),
            embedding_optimizer=embedding_optimizer.state_dict(),
            embedding_scheduler=embedding_scheduler.state_dict(),
            forward_net=forward_net.module.state_dict(),
            forward_optimizer=forward_optimizer.state_dict(),
            forward_scheduler=forward_scheduler.state_dict(),
            inverse_net=inverse_net.module.state_dict(),
            inverse_optimizer=inverse_optimizer.state_dict(),
            inverse_scheduler=inverse_scheduler.state_dict(),
            steps_processed=steps_processed,
            parameter_updates=parameter_updates,
            samples_collected=samples_collected,
            iteration=current_iteration
            ), 'checkpoint.pt')
        logger.info("done")
    elif int(os.environ['LOCAL_RANK']) == 0:
        # Checkpoint model params only
        logger.info("Saving initial checkpoint...")
        torch.save(dict(
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            embedding_optimizer=embedding_optimizer.state_dict(),
            embedding_scheduler=embedding_scheduler.state_dict(),
            forward_optimizer=forward_optimizer.state_dict(),
            forward_scheduler=forward_scheduler.state_dict(),
            inverse_optimizer=inverse_optimizer.state_dict(),
            inverse_scheduler=inverse_scheduler.state_dict(),
            iteration=current_iteration
            ), 'checkpoint.pt')
        logger.info("done")

    # Prep the queue and start processes
    for i in range(cfg.training.num_buffers):
        logger.info(f"Prefilling buffer {i}")
        learned_queue.put(i)

    learner_thread = threading.Thread(
            target=learn,
            name="learner",
            args=(),
            daemon=True
    )
    learner_thread.start()
    has_reset_timer = False
    best_mean_ep_return = -math.inf

    for iteration in it.count():
        buffer_index = learned_queue.get()
        if buffer_index == -1:
            logger.error("Error in learner thread caught! Aborting...")
            raise Exception("Error in learner thread")
        bsize, sampled_buffers = replay_buffer.sample_batch(B, 'actor', 8, max_remote_nodes=32)
        if bsize is not B:
            logger.error(f"Sampled batched size ({bsize}) is higher than B ({B})")
            sampled_buffers = sampled_buffers[B:]
        batch = {
            key: torch.stack([sampled_buffers[m][key] for m in range(B)], dim=1)
            for key in buffer_keys
        }
        initial_agent_state = tuple((
            torch.cat(ts, dim=1)
            for ts in zip(
                *[sampled_buffers[m]["initial_agent_state"] for m in range(B)]
            )
        ))
        # Fill buffer
        for state_index in range(len(initial_agent_state)):
            initial_agent_state_buffers[buffer_index][state_index][...] = initial_agent_state[state_index] 
        for k in buffer_keys:
            buffers[k][buffer_index] = batch[k]
        # Submit filled buffer to queue
        full_queue.put(buffer_index)

        if len(stats.keys()) > 0:
            current_iteration += 1
            if not has_reset_timer:
                impala_group.registered_sync("reset_learning_timer")
                has_reset_timer = True
            logger.info(stats)
            total_loss = stats["total_loss"]
            pg_loss = stats["pg_loss"]
            baseline_loss = stats["baseline_loss"]
            entropy_loss = stats["entropy_loss"]
            mean_ep_returns_moving = stats["mean_ep_returns_moving"]
            mean_abs_ep_return = stats["mean_abs_ep_return"]
            var_ep_returns_moving = stats["var_ep_returns_moving"]
            forward_loss = stats["forward_loss"]
            inverse_loss = stats["inverse_loss"]
            mean_intrinsic_reward = stats["mean_intrinsic_reward"]
            total_ep = stats["total_ep"]
            if is_leader:
                global_step = current_iteration
                writer.add_scalar("Training/TotalLoss", total_loss, global_step=global_step)
                writer.add_scalar("Training/PGLoss", pg_loss, global_step=global_step)
                writer.add_scalar("Training/BaselineLoss", baseline_loss, global_step=global_step)
                writer.add_scalar("Training/EntropyLoss", entropy_loss, global_step=global_step)
                writer.add_scalar("Training/ForwardLoss", forward_loss, global_step=global_step)
                writer.add_scalar("Training/InverseLoss", inverse_loss, global_step=global_step)
                writer.add_scalar("Training/MeanEpisodeReturns", mean_ep_returns_moving, global_step=global_step)
                writer.add_scalar("Training/MeanIntrinsicReward", mean_intrinsic_reward, global_step=global_step)
                writer.add_scalar("Training/MeanAbsEpisodeReturns", mean_abs_ep_return, global_step=global_step)
                writer.add_scalar("Training/VarEpisodeReturns", var_ep_returns_moving, global_step=global_step)
                writer.add_scalar("Training/TotalEpisodes", total_ep, global_step=global_step)
                if mean_ep_returns_moving > best_mean_ep_return:
                    logger.info("Saving best checkpoint...")
                    torch.save(dict(
                        optimizer=optimizer.state_dict(),
                        model=model.module.state_dict(),
                        scheduler=scheduler.state_dict(),
                        embedding_net=embedding_net.module.state_dict(),
                        embedding_optimizer=embedding_optimizer.state_dict(),
                        embedding_scheduler=embedding_scheduler.state_dict(),
                        forward_net=forward_net.module.state_dict(),
                        forward_optimizer=forward_optimizer.state_dict(),
                        forward_scheduler=forward_scheduler.state_dict(),
                        inverse_net=inverse_net.module.state_dict(),
                        inverse_optimizer=inverse_optimizer.state_dict(),
                        inverse_scheduler=inverse_scheduler.state_dict(),
                        steps_processed=steps_processed,
                        parameter_updates=parameter_updates,
                        samples_collected=samples_collected,
                        iteration=current_iteration
                        ), 'best.pt')
                    best_mean_ep_return = mean_ep_returns_moving
                    logger.info("done")

            if iteration % 20 == 0:
                if is_leader:
                    # A ring reduce is only one parameter update. We only process more steps and get a richer gradient
                    impala_group.registered_sync("increment_parameter_updates")
                impala_group.registered_sync("increment_step_processed")
                global_timer = impala_group.registered_sync("get_global_timer")
                learning_timer = impala_group.registered_sync("get_learning_timer")
                samples_collected = impala_group.registered_sync("get_samples_collected")
                parameter_updates = impala_group.registered_sync("get_parameter_updates")
                steps_processed = impala_group.registered_sync("get_step_processed")
                steps_sampled_s = ( 
                                samples_collected
                                / global_timer
                )
                param_updates_s = ( 
                                parameter_updates
                                / learning_timer
                )
                steps_processed_s = ( 
                                steps_processed
                                / learning_timer
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
                if is_leader:
                    global_step = current_iteration
                    writer.add_scalar("Speed/StepSampledS", steps_sampled_s, global_step=global_step)
                    writer.add_scalar("Speed/ParamUpdatesS", param_updates_s, global_step=global_step)
                    writer.add_scalar("Speed/StepProcessedS", steps_processed_s, global_step=global_step)

                    writer.add_scalar("Progress/StepSampled", samples_collected, global_step=global_step)
                    writer.add_scalar("Progress/ParamUpdates", parameter_updates, global_step=global_step)
                    writer.add_scalar("Progress/StepProcessed", steps_processed, global_step=global_step)
                    server.push(model.module, pull_on_fail=False)
                if (
                    is_leader and
                    steps_processed
                    > cfg.training.total_steps
                ):
                    impala_group.registered_sync("turn_global_switch_off")
                    break
                elif impala_group.registered_sync("get_global_switch") is False:
                    break
                if is_leader and parameter_updates % cfg.training.checkpointing_frequency == 0:
                    # Checkpoint
                    # - Model params
                    # - Optimizer state dict
                    # - steps_processed
                    # - param_parameter_updates
                    # - samples_collected
                    logger.info("Saving checkpoint...")
                    torch.save(dict(
                        model=model.module.state_dict(),
                        optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(),
                        embedding_net=embedding_net.module.state_dict(),
                        embedding_optimizer=embedding_optimizer.state_dict(),
                        embedding_scheduler=embedding_scheduler.state_dict(),
                        forward_net=forward_net.module.state_dict(),
                        forward_optimizer=forward_optimizer.state_dict(),
                        forward_scheduler=forward_scheduler.state_dict(),
                        inverse_net=inverse_net.module.state_dict(),
                        inverse_optimizer=inverse_optimizer.state_dict(),
                        inverse_scheduler=inverse_scheduler.state_dict(),
                        steps_processed=steps_processed,
                        parameter_updates=parameter_updates,
                        samples_collected=samples_collected,
                        iteration=current_iteration
                        ), 'checkpoint.pt')
                    logger.info("done")

                # This is a hack for now. We don't want two learner processes on the same machine savign simultaneously
                elif parameter_updates % cfg.training.checkpointing_frequency == 0 and int(os.environ['LOCAL_RANK']) == 0:
                    # Checkpoint model params only
                    logger.info("Saving checkpoint...")
                    torch.save(dict(
                        scheduler=scheduler.state_dict(),
                        optimizer=optimizer.state_dict(),
                        embedding_optimizer=embedding_optimizer.state_dict(),
                        embedding_scheduler=embedding_scheduler.state_dict(),
                        forward_optimizer=forward_optimizer.state_dict(),
                        forward_scheduler=forward_scheduler.state_dict(),
                        inverse_optimizer=inverse_optimizer.state_dict(),
                        inverse_scheduler=inverse_scheduler.state_dict(),
                        iteration=current_iteration
                        ), 'checkpoint.pt')
                    logger.info("done")
    logger.info("Left the main loop")
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
    env = create_env(cfg.env.task)
    observation_space = env.observation_space
    action_space = env.action_space
    env = ResettingEnvironment(env)
    model = Net(observation_space, action_space.n, cfg.model.embedding_size, cfg.model.lstm)
    env_output = env.initial()
    agent_state = model.initial_state(batch_size=1)
    agent_output, unused_state = model(env_output, agent_state)
    impala_group.barrier(is_leader)
    prof = glm.utils.SimpleProfiler()
    server.pull(model)
    buffers = _create_buffers(
        cfg.training.unroll_length, observation_space, action_space.n, agent_state
    )
    episode_state_count_dict = {}
    impala_group.registered_sync("reset_global_timer")
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
            episode_state_key = tuple(env_output['frame'].view(-1).tolist())
            if episode_state_key in episode_state_count_dict:
                episode_state_count_dict[episode_state_key] += 1
            else:
                episode_state_count_dict.update({episode_state_key: 1})
            buffers["episode_state_count"][0, ...] = torch.tensor(1 / np.sqrt(episode_state_count_dict.get(episode_state_key)))

        if env_output['done'][0][0]:
            episode_state_count_dict = {}

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

            episode_state_key = tuple(env_output['frame'].view(-1).tolist())
            if episode_state_key in episode_state_count_dict:
                episode_state_count_dict[episode_state_key] += 1
            else:
                episode_state_count_dict.update({episode_state_key: 1})
            buffers["episode_state_count"][t+ 1, ...] = torch.tensor(1 / np.sqrt(episode_state_count_dict.get(episode_state_key)))

            if env_output['done'][0][0]:
                episode_state_count_dict = {}

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


def evaluator(
    init: Tuple[
        glm.distributed.World,
        glm.distributed.RpcGroup,
        glm.buffers.DistributedBuffer,
        glm.servers.PushPullModelServer,
    ],
    cfg: DictConfig,
):
    torch.set_num_threads(16)
    logger = glm.utils.default_logger
    logger.info("Booting an Impala evaluator")
    world, impala_group, replay_buffer, server = init
    is_leader = world.rank == 0
    logger.info("My friends: " + str(world.get_members()))
    env = create_env(cfg.env.task)
    observation_space = env.observation_space
    action_space = env.action_space
    model = Net(observation_space, action_space.n, cfg.model.embedding_size, cfg.model.lstm)
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    model = model.to(device)
    logger.info("Device: " + str(device))
    env = ResettingEnvironment(env)
    env_output = env.initial()
    agent_state = model.initial_state(batch_size=1, device=device)
    env_output = {
            k: t.to(device=device) for k, t in env_output.items()
    }
    agent_output, unused_state = model(env_output, agent_state)
    impala_group.barrier(is_leader)
    prof = glm.utils.SimpleProfiler()
    done = env_output["done"]
    best_mean_returns = -math.inf
    for iteration in it.count():
        server.pull(model)
        model = model.to(device)
        #model.eval()
        returns = []
        steps_of_episodes = []
        # Do new rollout.
        for t in range(cfg.evaluation.num_episodes):
            while not done:
                with torch.no_grad():
                    env_output = {
                        k: t.to(device=device, non_blocking=True) for k, t in env_output.items()
                    }
                    agent_output, agent_state = model(env_output, agent_state)
                env_output = env.step(agent_output["action"].cpu())
                done = env_output["done"].item()
            returns.append(env_output["episode_return"].item())
            steps_of_episodes.append(env_output["episode_step"].item())
            done = False
            if impala_group.registered_sync("get_global_switch") is False:
                break

        returns = torch.Tensor(returns)
        steps_of_episodes = torch.Tensor(steps_of_episodes)
        mean_returns = torch.mean(returns)
        var_returns = torch.var(returns)
        mean_steps = torch.mean(steps_of_episodes)
        var_steps = torch.var(steps_of_episodes)
        logger.info(dict(returns=returns, steps_of_episodes=steps_of_episodes))
        results = dict(
            mean_returns=mean_returns,
            var_returns=var_returns,
            mean_steps=mean_steps,
            var_steps=var_steps
        )
        logger.info(results)

        impala_group.registered_sync("log_evaluator_results", args=(results,))
        if mean_returns > best_mean_returns:
            best_mean_returns = mean_returns
            impala_group.registered_sync("save_evaluator_model", args=(model.to(torch.device('cpu')).state_dict(), best_mean_returns))
        if impala_group.registered_sync("get_global_switch") is False:
            break

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
        impala_group.register("set_samples_collected", sample_counter.set)

        parameter_update_counter = glm.widgets.Counter(step=20)
        impala_group.register(
            "increment_parameter_updates", parameter_update_counter.count
        )
        impala_group.register("get_parameter_updates", parameter_update_counter.get)
        impala_group.register("set_parameter_updates", parameter_update_counter.set)

        step_processed_counter = glm.widgets.Counter(
            step=cfg.training.batch_size * cfg.training.unroll_length * 20
        )
        impala_group.register("increment_step_processed", step_processed_counter.count)
        impala_group.register("get_step_processed", step_processed_counter.get)
        impala_group.register("set_step_processed", step_processed_counter.set)

        global_switch = glm.widgets.Switch(state=True)
        impala_group.register("turn_global_switch_off", global_switch.off)
        impala_group.register("get_global_switch", global_switch.get)

        global_timer = glm.widgets.Timer()
        impala_group.register("get_global_timer", global_timer.end)
        impala_group.register("reset_global_timer", global_timer.begin)

        learning_timer = glm.widgets.Timer()
        impala_group.register("get_learning_timer", learning_timer.end)
        impala_group.register("reset_learning_timer", learning_timer.begin)
    return (world, impala_group, replay_buffer, server)


def evaluator_script(cfg: DictConfig):
    logger = glm.utils.default_logger
    torch.set_num_threads(4)
    logger.info("Evaluator Script")
    logger.info("Script: " + cfg.env.task)
    env = create_env(cfg.env.task)
    observation_space = env.observation_space
    action_space = env.action_space
    model = Net(observation_space, action_space.n, cfg.model.embedding_size, cfg.model.lstm)
    #chkpt = torch.load("best_eval.pt")
    chkpt = torch.load("best.pt")
    model.load_state_dict(chkpt["model"])
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    model = model.to(device)
    logger.info("Device: " + str(device))
    env = ResettingEnvironment(env)
    env_output = env.initial()
    agent_state = model.initial_state(batch_size=1, device=device)
    env_output = {
            k: t.to(device=device) for k, t in env_output.items()
    }
    agent_output, unused_state = model(env_output, agent_state)
    prof = glm.utils.SimpleProfiler()
    done = env_output["done"]
    for iteration in range(10):
        #model.eval()
        returns = []
        steps_of_episodes = []
        # Do new rollout.
        for t in range(cfg.evaluation.num_episodes):
            while not done:
                with torch.no_grad():
                    env_output = {
                            k: t.to(device=device, non_blocking=True) for k, t in env_output.items()
                    }
                    agent_output, agent_state = model(env_output, agent_state)
                env_output = env.step(agent_output["action"].cpu())
                done = env_output["done"].item()
            returns.append(env_output["episode_return"].item())
            steps_of_episodes.append(env_output["episode_step"].item())
            done = False

        returns = torch.Tensor(returns)
        steps_of_episodes = torch.Tensor(steps_of_episodes)
        mean_returns = torch.mean(returns)
        var_returns = torch.var(returns)
        mean_steps = torch.mean(steps_of_episodes)
        var_steps = torch.var(steps_of_episodes)
        logger.info(dict(returns=returns, steps_of_episodes=steps_of_episodes))
        results = dict(
            mean_returns=mean_returns,
            var_returns=var_returns,
            mean_steps=mean_steps,
            var_steps=var_steps
        )
        logger.info(results)



hydra.main()(glm.launcher.launch(init, {"actor": actor, "learner": learner, "evaluator": evaluator}, {"evaluate": evaluator_script}))()
