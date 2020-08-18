import numpy as np

from skmultiflow.trees._split_criterion import SplitCriterion


class InfoGainExpandCriterion(SplitCriterion):
    """ Information Gain rule split criterion.

    A measure of how often a randomly chosen element from the set would be incorrectly labeled if it was randomly
    labeled according to the distribution of labels in the subset.

    This implementation is specific to rule-based methods.

    """
    def __init__(self, min_branch_frac_option=0.01):
        super().__init__()
        # Minimum fraction of weight required down at least two branches.
        self.min_branch_frac_option = min_branch_frac_option
        self.lowest_entropy = None
        self.best_idx = 0

    def get_merit_of_split(self, pre_split_dist, post_split_dist):
        if self.num_subsets_greater_than_frac(post_split_dist, self.min_branch_frac_option) < 2:
            return -np.inf
        return self.compute_entropy(pre_split_dist) - self.compute_entropy(post_split_dist)

    @staticmethod
    def get_range_of_merit(pre_split_dist):
        num_classes = len(pre_split_dist)
        num_classes = num_classes if num_classes > 2 else 2
        return np.log2(num_classes)

    def compute_entropy(self, dist):
        if isinstance(dist, dict):
            return self._compute_entropy_dict(dist)
        elif isinstance(dist, list):
            return self._compute_entropy_list(dist)

    @staticmethod
    def _compute_entropy_dict(dist):
        entropy = 0.0
        dis_sums = 0.0
        for _, d in dist.items():
            if d > 0.0:
                entropy -= d * np.log2(d)
                dis_sums += d
        return (entropy + dis_sums * np.log2(dis_sums)) / dis_sums if dis_sums > 0.0 else 0.0

    def _compute_entropy_list(self, dists):
        total_weight = 0.0
        dist_weights = [0.0] * len(dists)
        for i in range(len(dists)):
            dist_weights[i] = sum(dists[i].values())
            total_weight += dist_weights[i]
        entropy = 0.0
        for i in range(len(dists)):
            _entropy = self.compute_entropy(dists[i])
            if self.lowest_entropy is None or _entropy < self.lowest_entropy:
                self.lowest_entropy = _entropy
                self.best_idx = i
            entropy += dist_weights[i] * _entropy
        return entropy / total_weight

    @staticmethod
    def num_subsets_greater_than_frac(distributions, min_frac):
        total_weight = 0.0
        dist_sums = [0.0] * len(distributions)
        for i in range(len(dist_sums)):
            dist_sums[i] = sum(distributions[i].values())
            total_weight += dist_sums[i]
        num_greater = 0
        for d in dist_sums:
            if (d / total_weight) > min_frac:
                num_greater += 1
        return num_greater
