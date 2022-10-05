import dataclasses
import random
import gym


@dataclasses.dataclass
class VendingMachine:
    threshold: float
    attempts: int = 0
    successes: int = 0


class CandyCaneContest(gym.Env):
    """

    Examples
    --------

    >>> import gym
    >>> from river import bandit
    >>> from river import stats

    >>> env = gym.make(
    ...     'river_bandits/CandyCaneContest-v0'
    ... )
    >>> _ = env.reset(seed=42)
    >>> _ = env.action_space.seed(123)

    >>> metric = stats.Sum()
    >>> while True:
    ...      action = env.action_space.sample()
    ...      observation, reward, terminated, truncated, info = env.step(action)
    ...      metric = metric.update(reward)
    ...      if terminated or truncated:
    ...         break

    >>> metric
    Sum: 734.

    """
    n_rounds = 2000

    def __init__(self, n_machines=100, reward_decay=0.03):
        self.n_machines = n_machines
        self.reward_decay = reward_decay

        self.action_space = gym.spaces.Discrete(n_machines)
        self.observation_space = gym.spaces.Dict({
            "attempts": gym.spaces.Tuple([gym.spaces.Discrete(self.n_rounds)] * n_machines),
            "successes": gym.spaces.Tuple([gym.spaces.Discrete(self.n_rounds)] * n_machines),
        })
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
            VendingMachine(
                threshold=threshold
            )
            for threshold in self.np_random.uniform(size=self.n_machines)
        ]
        observation = self._get_observation()
        info = self._get_info()
        return observation, info

    def step(self, action):
        machine = self._machines[action]
        reward = float(self.np_random.uniform() < machine.threshold)
        machine.threshold *= (1 - self.reward_decay)

        observation = self._get_observation()
        info = self._get_info()
        terminated = False
        truncated = False
        return observation, reward, terminated, truncated, info
