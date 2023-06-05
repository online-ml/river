from __future__ import annotations

import math
import random

from river import bandit


class EpsilonGreedy(bandit.base.Policy):
    r"""$\varepsilon$-greedy bandit policy.

    Performs arm selection by using an $\varepsilon$-greedy bandit strategy. An arm is selected at each
    step. The best arm is selected (1 - $\varepsilon$)% of the time.

    Selection bias is a common problem when using bandits. This bias can be mitigated by using
    burn-in phase. Each model is given the chance to learn during the first `burn_in` steps.

    Parameters
    ----------
    epsilon
        The probability of exploring.
    decay
        The decay rate of epsilon.
    reward_obj
        The reward object used to measure the performance of each arm. This can be a metric, a
        statistic, or a distribution.
    burn_in
        The number of steps to use for the burn-in phase. Each arm is given the chance to be pulled
        during the burn-in phase. This is useful to mitigate selection bias.
    seed
        Random number generator seed for reproducibility.

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

    >>> policy = bandit.EpsilonGreedy(epsilon=0.9, seed=101)

    >>> metric = stats.Sum()
    >>> while True:
    ...     arm = policy.pull(range(env.action_space.n))
    ...     observation, reward, terminated, truncated, info = env.step(arm)
    ...     policy = policy.update(arm, reward)
    ...     metric = metric.update(reward)
    ...     if terminated or truncated:
    ...         break

    >>> metric
    Sum: 775.

    References
    ----------
    [^1]: [ε-Greedy Algorithm - The Multi-Armed Bandit Problem and Its Solutions - Lilian Weng](https://lilianweng.github.io/lil-log/2018/01/23/the-multi-armed-bandit-problem-and-its-solutions.html#%CE%B5-greedy-algorithm)

    """

    def __init__(
        self, epsilon: float, decay=0.0, reward_obj=None, burn_in=0, seed: int | None = None
    ):
        super().__init__(reward_obj=reward_obj, burn_in=burn_in)
        self.epsilon = epsilon
        self.decay = decay
        self.seed = seed
        self._rng = random.Random(seed)

    @property
    def current_epsilon(self):
        """The value of epsilon after factoring in the decay rate."""
        if self.decay:
            return self.epsilon * math.exp(-self._n * self.decay)
        return self.epsilon

    def _pull(self, arm_ids):
        return (
            self._rng.choice(arm_ids)  # explore
            if self._rng.uniform(0, 1) < self.current_epsilon
            else max(arm_ids, key=lambda arm: self._rewards[arm].get())  # exploit
        )

    @classmethod
    def _unit_test_params(cls):
        yield {"epsilon": 0.2}
