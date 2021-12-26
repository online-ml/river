import math
import operator

from .base import BanditPolicy


class UCB(BanditPolicy):
    def __init__(self, delta, burn_in, seed):
        super().__init__(burn_in, seed)
        self.delta = delta

    def _pull(self, bandit):
        sign = operator.pos if bandit.metric.bigger_is_better else operator.neg
        upper_bounds = [
            sign(arm.metric.get())
            + self.delta * math.sqrt(2 * math.log(bandit.n_pulls) / arm.n_pulls)
            if arm.n_pulls
            else math.inf
            for arm in bandit.arms
        ]
        yield max(bandit.arms, key=lambda arm: upper_bounds[arm.index])
