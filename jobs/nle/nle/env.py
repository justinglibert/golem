import torch
import gym
import nle
import time


def _format_observations(observation, keys=("glyphs", "blstats")):
    observations = {}
    for key in keys:
        entry = observation[key]
        entry = torch.from_numpy(entry)
        entry = entry.view((1, 1) + entry.shape)  # (...) -> (T,B,...).
        observations[key] = entry
    return observations


class ResettingEnvironment:
    """Turns a Gym environment into something that can be step()ed indefinitely."""

    def __init__(self, gym_env):
        self.gym_env = gym_env
        self.episode_return = None
        self.episode_step = None

    def initial(self):
        initial_reward = torch.zeros(1, 1)
        # This supports only single-tensor actions ATM.
        initial_last_action = torch.zeros(1, 1, dtype=torch.int64)
        self.episode_return = torch.zeros(1, 1)
        self.episode_step = torch.zeros(1, 1, dtype=torch.int32)
        initial_done = torch.ones(1, 1, dtype=torch.uint8)

        result = _format_observations(self.gym_env.reset())
        result.update(
            reward=initial_reward,
            done=initial_done,
            episode_return=self.episode_return,
            episode_step=self.episode_step,
            last_action=initial_last_action,
        )
        return result

    def step(self, action):
        observation, reward, done, unused_info = self.gym_env.step(
            action.item())
        self.episode_step += 1
        self.episode_return += reward
        episode_step = self.episode_step
        episode_return = self.episode_return
        if done:
            observation = self.gym_env.reset()
            self.episode_return = torch.zeros(1, 1)
            self.episode_step = torch.zeros(1, 1, dtype=torch.int32)

        result = _format_observations(observation)

        reward = torch.tensor(reward).view(1, 1)
        done = torch.tensor(done).view(1, 1)

        result.update(
            reward=reward,
            done=done,
            episode_return=episode_return,
            episode_step=episode_step,
            last_action=action,
        )
        return result

    def close(self):
        self.gym_env.close()


def create_env(name, *args, **kwargs):
    return gym.make(name, observation_keys=("glyphs", "blstats"), *args, **kwargs)
