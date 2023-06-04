from __future__ import annotations

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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._actual_rewards = self.np_random.normal(loc=0, scale=1, size=self.k).tolist()
        self._best_arm = max(enumerate(self._actual_rewards), key=lambda x: x[1])[0]
        observation = self._best_arm
        info = {}
        return observation, info

    def step(self, arm):
        arm_reward = self._actual_rewards[arm]
        reward = self.np_random.normal(loc=arm_reward, scale=1)

        observation = self._best_arm
        info = {}
        terminated = False
        truncated = False
        return observation, reward, terminated, truncated, info
