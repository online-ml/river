from __future__ import annotations

from .base import SplitCriterion


class VarianceRatioSplitCriterion(SplitCriterion):
    """Variance Ratio split criterion.

    Parameters
    ----------
    min_samples_split
        The minimum number of samples per post split dist element.

    """

    def __init__(self, min_samples_split: int = 5):
        super().__init__()
        self.min_samples_split = min_samples_split

    def merit_of_split(self, pre_split_dist, post_split_dist):
        vr = 0
        n = pre_split_dist.mean.n

        count = 0
        for i in range(len(post_split_dist)):
            n_i = post_split_dist[i].mean.n
            if n_i >= self.min_samples_split:
                count += 1
        if count == len(post_split_dist):
            vr = 1
            var = self.compute_var(pre_split_dist)
            for i in range(len(post_split_dist)):
                n_i = post_split_dist[i].mean.n
                vr -= (n_i / n) * (self.compute_var(post_split_dist[i]) / var)
        return vr

    @staticmethod
    def compute_var(dist):
        return dist.get()

    @staticmethod
    def range_of_merit(pre_split_dist):
        return 1.0

    @staticmethod
    def select_best_branch(children_stats):
        n0 = children_stats[0].mean.n
        n1 = children_stats[1].mean.n

        n = n0 + n1

        vr0 = (n0 / n) * VarianceRatioSplitCriterion.compute_var(children_stats[0])
        vr1 = (n1 / n) * VarianceRatioSplitCriterion.compute_var(children_stats[1])

        # Return the branch that most minimizes the variance
        return 0 if vr0 <= vr1 else 1
