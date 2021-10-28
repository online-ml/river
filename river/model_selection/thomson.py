import math

from .bandit import BanditPolicy


class ThompsonSampler(BanditPolicy):
    def __init__(self, burn_in, seed):
        super().__init__(burn_in, seed)

    def _pull(self, bandit):
        yield (
            self.rng.choice(bandit.arms)  # explore
            if self.rng.random() < self.current_epsilon(n=bandit.n_pulls)
            else bandit.best_arm  # exploit
        )
