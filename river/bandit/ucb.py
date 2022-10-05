import math
import operator

from river import bandit


class UCB(bandit.base.BanditPolicy):
    """

    References
    ----------
    [^1]: [Lai, T. L., & Robbins, H. (1985). Asymptotically efficient adaptive allocation rules. Advances in applied mathematics, 6(1), 4-22.](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.674.1620&rep=rep1&type=pdf)
    [^2]: [Upper Confidence Bounds - The Multi-Armed Bandit Problem and Its Solutions - Lilian Weng](https://lilianweng.github.io/lil-log/2018/01/23/the-multi-armed-bandit-problem-and-its-solutions.html#upper-confidence-bounds)
    [^3]: [The Upper Confidence Bound Algorithm - Bandit Algorithms](https://banditalgs.com/2016/09/18/the-upper-confidence-bound-algorithm/)

    """
    def __init__(self, delta, reward_obj=None, seed=None):
        super().__init__(reward_obj, seed)
        self.delta = delta

    # def pull(self, arms):
    #     sign = operator.pos if bandit.metric.bigger_is_better else operator.neg
    #     upper_bounds = [
    #         sign(arm.metric.get())
    #         + self.delta * math.sqrt(2 * math.log(bandit.n_pulls) / arm.n_pulls)
    #         if arm.n_pulls
    #         else math.inf
    #         for arm in bandit.arms
    #     ]
    #     yield max(bandit.arms, key=lambda arm: upper_bounds[arm.index])
