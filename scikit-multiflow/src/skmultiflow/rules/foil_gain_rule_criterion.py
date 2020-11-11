import numpy as np

from skmultiflow.trees._split_criterion import SplitCriterion


class FoilGainExpandCriterion(SplitCriterion):
    """ FOIL's Information Gain rule split criterion.

    A measure similar to Information Gain, used in the first-order inductive learner (FOIL) algorithm.

    This implementation is specific to rule-based methods.

    """
    def __init__(self, min_branch_frac_option=0.01):
        super().__init__()
        # Minimum fraction of weight required down at least two branches.
        self.min_branch_frac_option = min_branch_frac_option
        self.best_idx = 0
        self.class_idx = 0

    def get_merit_of_split(self, pre_split_dist, post_split_dist):
        if self.num_subsets_greater_than_frac(post_split_dist, self.min_branch_frac_option) < 2:
            return -np.inf
        entropy1 = self.compute_entropy(post_split_dist[0])
        entropy2 = self.compute_entropy(post_split_dist[1])
        if entropy1 >= entropy2:
            self.best_idx = 0
        else:
            self.best_idx = 1
        entropy = min(entropy1, entropy2)
        gain = entropy - self.compute_entropy(pre_split_dist)
        try:
            return post_split_dist[self.best_idx][self.class_idx] * gain
        except KeyError:
            return 0

    def get_range_of_merit(self, pre_split_dist):
        num_classes = len(pre_split_dist)
        # num_classes = num_classes if num_classes > 2 else 2
        return np.log2(num_classes)
        # return  1
        # return -np.log2(pre_split_dist[self.class_idx]/sum(pre_split_dist.values())) * pre_split_dist[self.class_idx]

    def compute_entropy(self, dist):
        try:
            return np.log2(dist[self.class_idx] / sum(dist.values()))
        except KeyError:
            return 0

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
