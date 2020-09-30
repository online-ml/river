from .variance_reduction_split_criterion import VarianceReductionSplitCriterion


# This class extends VarianceReductionSplitCriterion since it just computes
# the variance differently than its ancestor (considering multiple targets)
class IntraClusterVarianceReductionSplitCriterion(VarianceReductionSplitCriterion):

    def __init__(self, min_samples_split: int = 5):
        super().__init__(min_samples_split)

    def get_merit_of_split(self, pre_split_dist, post_split_dist):
        icvr = 0.0
        n = pre_split_dist[0]

        count = 0

        for dist in post_split_dist:
            n_i = dist[0]
            if n_i >= self.min_samples_split:
                count += 1

        if count == len(post_split_dist):
            icvr = self.compute_var(pre_split_dist)
            for dist in post_split_dist:
                n_i = dist[0]
                icvr -= n_i/n * self.compute_var(dist)
        return icvr

    @staticmethod
    def compute_var(dist):
        n = dist[0]
        sum_ = dist[1]
        sum_sq = dist[2]

        icvr = sum((sum_sq - ((sum_ * sum_) / n)).values()) / (n - 1)
        return icvr if icvr > 0 else 0.
