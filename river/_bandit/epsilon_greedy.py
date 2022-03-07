import math

from .base import BanditPolicy


class EpsilonGreedy(BanditPolicy):
    r"""$\eps$-greedy strategy."""

    def __init__(self, epsilon: float, decay: float, burn_in, seed):
        super().__init__(burn_in, seed)
        self.epsilon = epsilon
        self.decay = decay

    def current_epsilon(self, n: int):
        if self.decay:
            return self.epsilon * math.exp(-n * self.decay)
        return self.epsilon

    def _pull(self, bandit):
        yield (
            self.rng.choice(bandit.arms)  # explore
            if self.rng.random() < self.current_epsilon(n=bandit.n_pulls)
            else bandit.best_arm  # exploit
        )
