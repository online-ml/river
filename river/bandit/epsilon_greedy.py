import math
from river import bandit


class EpsilonGreedy(bandit.base.Policy):
    r"""$\eps$-greedy bandit policy.

    Performs arm selection by using an $\eps$-greedy bandit strategy. An arm is selected at each
    step. The best arm is selected (1 - $\eps$%) of the time.

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

    References
    ----------
    [^1]: [ε-Greedy Algorithm - The Multi-Armed Bandit Problem and Its Solutions - Lilian Weng](https://lilianweng.github.io/lil-log/2018/01/23/the-multi-armed-bandit-problem-and-its-solutions.html#%CE%B5-greedy-algorithm)

    """

    def __init__(self, epsilon: float, decay: float, reward_obj=None, burn_in=0, seed = None):
        super().__init__(reward_obj, burn_in, seed)
        self.epsilon = epsilon
        self.decay = decay

    @property
    def current_epsilon(self):
        """The value of epsilon after factoring in the decay rate."""
        if self.decay:
            return self.epsilon * math.exp(-self._n * self.decay)
        return self.epsilon

    def _pull(self, arms):
        return (
            self.rng.choice(arms)  # explore
            if self.best_arm is None or self.rng.random() < self.current_epsilon
            else self.best_arm  # exploit
        )

    @classmethod
    def _unit_test_params(cls):
        yield {"epsilon": 0.2, "decay": 0.0}
