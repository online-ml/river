import math
from river import bandit


class EpsilonGreedy(bandit.base.BanditPolicy):
    r"""$\eps$-greedy strategy.


    References
    ----------
    [^1]: [ε-Greedy Algorithm - The Multi-Armed Bandit Problem and Its Solutions - Lilian Weng](https://lilianweng.github.io/lil-log/2018/01/23/the-multi-armed-bandit-problem-and-its-solutions.html#%CE%B5-greedy-algorithm)

    """

    def __init__(self, epsilon: float, decay: float, reward_obj=None, seed = None):
        super().__init__(reward_obj, seed)
        self.epsilon = epsilon
        self.decay = decay
        self._n = 0

    @property
    def current_epsilon(self):
        if self.decay:
            return self.epsilon * math.exp(-self._n * self.decay)
        return self.epsilon

    def pull(self, arms):
        yield (
            self.rng.choice(arms)  # explore
            if self.best_arm is None or self.rng.random() < self.current_epsilon
            else self.best_arm  # exploit
        )

    def update(self, arm, *reward_args, **reward_kwargs):
        super().update(arm, *reward_args, **reward_kwargs)
        self._n += 1
