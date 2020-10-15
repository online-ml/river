from .base_split_criterion import SplitCriterion


class VarianceReductionSplitCriterion(SplitCriterion):
    """Variance Reduction split criterion.

    Often employed in cases where the target variable is continuous (regression tree),
    meaning that use of many other metrics would first require discretization before being applied.

    `Wikipedia entry <https://en.wikipedia.org/wiki/Decision_tree_learning#Variance_reduction>`_

    """

    def __init__(self, min_samples_split: int = 5):
        super().__init__()
        self.min_samples_split = min_samples_split

    def get_merit_of_split(self, pre_split_dist, post_split_dist):
        vr = 0.0
        n = pre_split_dist[0]

        count = 0
        for i in range(len(post_split_dist)):
            n_i = post_split_dist[i][0]
            if n_i >= self.min_samples_split:
                count += 1
        if count == len(post_split_dist):

            vr = self.compute_var(pre_split_dist)
            for i in range(len(post_split_dist)):
                n_i = post_split_dist[i][0]
                vr -= n_i / n * self.compute_var(post_split_dist[i])
        return vr

    @staticmethod
    def compute_var(dist):

        n = dist[0]
        sum_ = dist[1]
        sum_sq = dist[2]

        var = (sum_sq - (sum_ * sum_) / n) / (n - 1) if n > 1 else 0
        return var if var > 0.0 else 0.0

    @staticmethod
    def get_range_of_merit(pre_split_dist):
        # The VR values are unbounded, but as we compare the ratio between the attributes' VRs
        # the actual range is between 0 (the second best candidate has a merit of zero) and 1
        # (both compared split candidates have the same merit).
        return 1.0
