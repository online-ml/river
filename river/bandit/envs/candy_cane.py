from __future__ import annotations

import dataclasses

import gym


@dataclasses.dataclass
class VendingMachine:
    threshold: float
    attempts: int = 0
    successes: int = 0


class CandyCaneContest(gym.Env):
    """Candy cane contest Kaggle competition.

    Parameters
    ----------
    n_machines
        Number of vending machines.
    reward_decay
        The multiplicate rate at which the expected reward of each vending machine decays.

    Examples
    --------

    >>> import gym
    >>> from river import stats

    >>> env = gym.make('river_bandits/CandyCaneContest-v0')
    >>> _ = env.reset(seed=42)
    >>> _ = env.action_space.seed(123)

    >>> metric = stats.Sum()
    >>> while True:
    ...     arm = env.action_space.sample()
    ...     observation, reward, terminated, truncated, info = env.step(arm)
    ...     metric = metric.update(reward)
    ...     if terminated or truncated:
    ...         break

    >>> metric
    Sum: 734.

    References
    ----------
    [^1]: [Santa 2020 - The Candy Cane Contest](https://www.kaggle.com/c/santa-2020)

    """

    n_steps = 2000

    def __init__(self, n_machines=100, reward_decay=0.03):
        self.n_machines = n_machines
        self.reward_decay = reward_decay

        self.action_space = gym.spaces.Discrete(n_machines)
        self.observation_space = gym.spaces.Dict(
            {
                "attempts": gym.spaces.Tuple([gym.spaces.Discrete(self.n_steps)] * n_machines),
                "successes": gym.spaces.Tuple([gym.spaces.Discrete(self.n_steps)] * n_machines),
            }
        )
        self.reward_range = (0.0, 1.0)

    def _get_observation(self):
        return {
            "attempts": tuple(machine.attempts for machine in self._machines),
            "successes": tuple(machine.successes for machine in self._machines),
        }

    def _get_info(self):
        return {
            "thresholds": tuple(machine.threshold for machine in self._machines),
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._machines = [
            VendingMachine(threshold=threshold)
            for threshold in self.np_random.uniform(size=self.n_machines)
        ]
        observation = self._get_observation()
        info = self._get_info()
        return observation, info

    def step(self, machine_index):
        machine = self._machines[machine_index]
        reward = float(self.np_random.uniform() < machine.threshold)
        machine.threshold *= 1 - self.reward_decay

        observation = self._get_observation()
        info = self._get_info()
        terminated = False
        truncated = False
        return observation, reward, terminated, truncated, info
