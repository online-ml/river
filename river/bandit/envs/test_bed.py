import dataclasses
import math

import gym


class KArmedTestbed(gym.Env):
    """k-armed testbed.

    This is a simple environment that can be used to test bandit algorithms. It is based on the
    10 armed testbed described in the book "Reinforcement Learning: An Introduction" by Sutton and
    Barto.

    Parameters
    ----------
    k
        Number of arms.

    """

    n_steps = 1000

    def __init__(self, k: int = 10):
        super().__init__()
        self.k = k
        self.action_space = gym.spaces.Discrete(k)
        self.observation_space = gym.spaces.Discrete(k)
        self.reward_range = (-math.inf, math.inf)

    def _get_observation(self):
        return max(enumerate(self.actual_rewards), key=lambda x: x[1])[0]

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.actual_rewards = self.np_random.normal(loc=0, scale=1, size=self.k).tolist()
        observation = self._get_observation()
        info = self._get_info()
        return observation, info

    def step(self, action):
        arm_reward = self.actual_rewards[action]
        reward = arm_reward + self.np_random.normal(loc=0, scale=1)

        observation = self._get_observation()
        info = self._get_info()
        terminated = False
        truncated = False
        return observation, reward, terminated, truncated, info
