import hydra
from omegaconf import DictConfig
from typing import Tuple, List, Any
import golem as glm
import torch
import time
from .net import NetHackNet as Net
from .env import create_env, ResettingEnvironment


def learner(init: Tuple, cfg: DictConfig):
    logger = glm.utils.default_logger
    logger.info("I am the learner")


def create_buffers(unroll_length, observation_space, num_actions, model):
    # Get specimens to infer shapes and dtypes.
    size = (unroll_length + 1,)
    samples = {k: torch.from_numpy(v)
               for k, v in observation_space.sample().items()}

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
    state = model.initial_state(batch_size=1)
    buffers["initial_agent_state"] = state

    return buffers


def actor(init: Tuple[glm.distributed.World, glm.distributed.RpcGroup, glm.buffers.DistributedBuffer], cfg: DictConfig):
    torch.set_num_threads(4)
    logger = glm.utils.default_logger
    logger.info("Booting an NLE actor")
    world, impala_group, replay_buffer = init
    logger.info("My friends: " + str(world.get_members()))
    start = time.time()
    steps = 0
    env = create_env('NetHackScore-v0', savedir=None)
    observation_space = env.observation_space
    action_space = env.action_space
    env = ResettingEnvironment(env)
    model = Net(observation_space, action_space.n, True)
    buffers = create_buffers(
        80, observation_space, action_space.n, model)
    env_output = env.initial()
    agent_state = model.initial_state(batch_size=1)
    agent_output, unused_state = model(env_output, agent_state)
    impala_group.barrier()
    while impala_group.registered_sync("get_global_switch"):
        # Write old rollout end.
        for key in env_output:
            buffers[key][0, ...] = env_output[key]
        for key in agent_output:
            buffers[key][0, ...] = agent_output[key]
        for i, tensor in enumerate(agent_state):
            buffers["initial_agent_state"][i][...] = tensor

        # Do new rollout.
        for t in range(80):
            with torch.no_grad():
                agent_output, agent_state = model(env_output, agent_state)

            env_output = env.step(agent_output["action"])

            for key in env_output:
                buffers[key][t + 1, ...] = env_output[key]
            for key in agent_output:
                buffers[key][t + 1, ...] = agent_output[key]

        replay_buffer.append(buffers)
        impala_group.registered_async("increment_sample_collected")

        if world.rank == 0 and impala_group.registered_sync("get_samples_collected") > 20000:
            impala_group.registered_sync("turn_global_switch_off")

    print("done")
    for k in buffers.keys():
        print(k, buffers[k].shape if type(buffers[k])
              is torch.Tensor else len(buffers[k]))
    print(replay_buffer.size())
    print(replay_buffer.all_size())
    print(len(replay_buffer.sample_batch(2)))
    print(world.name, impala_group.registered_sync("get_samples_collected"))
    print("{} steps/s", impala_group.registered_sync("get_samples_collected") /
          impala_group.registered_sync("get_global_timer"))
    # Have a barrier at the end to make sure everything is sync before exiting
    impala_group.barrier()


def init(cfg: DictConfig):
    logger = glm.utils.default_logger
    world = glm.distributed.create_world_with_env()
    impala_group = world.create_rpc_group(
        "impala", world.get_members())
    replay_buffer = glm.buffers.DistributedBuffer(
        buffer_name="buffer", group=impala_group,
        buffer_size=200
    )
    # Counters and Switch
    if world.rank == 0:
        logger.info("I am the rank 0. Creating counters")
        sample_counter = glm.widgets.Counter(step=80)
        impala_group.register(
            "increment_sample_collected", sample_counter.count)
        impala_group.register("get_samples_collected", sample_counter.get)
        global_switch = glm.widgets.Switch(state=True)
        impala_group.register(
            "turn_global_switch_off", global_switch.off)
        impala_group.register("get_global_switch", global_switch.get)
        global_timer = glm.widgets.Timer()
        impala_group.register("get_global_timer", global_timer.end)
    return (world, impala_group, replay_buffer)


hydra.main()(glm.launcher.launch(init, {"actor": actor, "learner": learner}))()
