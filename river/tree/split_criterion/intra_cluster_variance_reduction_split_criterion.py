from __future__ import annotations

from .variance_reduction_split_criterion import VarianceReductionSplitCriterion


# This class extends VarianceReductionSplitCriterion since it just computes
# the variance differently than its ancestor (considering multiple targets)
class IntraClusterVarianceReductionSplitCriterion(VarianceReductionSplitCriterion):
    def __init__(self, min_samples_split: int = 5):
        super().__init__(min_samples_split)

    def merit_of_split(self, pre_split_dist, post_split_dist):
        icvr = 0.0
        n = list(pre_split_dist.values())[0].mean.n

        count = 0

        for dist in post_split_dist:
            n_i = list(dist.values())[0].mean.n
            if n_i >= self.min_samples_split:
                count += 1

        if count == len(post_split_dist):
            icvr = self.compute_var(pre_split_dist)
            for dist in post_split_dist:
                n_i = list(dist.values())[0].mean.n
                icvr -= n_i / n * self.compute_var(dist)
        return icvr

    @staticmethod
    def compute_var(dist):
        icvr = [vr.get() for vr in dist.values()]
        n = len(icvr)
        return sum(icvr) / n if n > 0 else 0.0
