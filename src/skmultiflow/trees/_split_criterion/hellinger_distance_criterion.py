import numpy as np

from .base_split_criterion import SplitCriterion


class HellingerDistanceCriterion(SplitCriterion):
    """ Hellinger Distance split criterion.

    The Hellinger distance is a measure of distributional divergence.
    It is used as the splitting criterion [1]_ on decision trees to to address
    the imbalanced data problem.

    References
    ----------
    .. [1] Cieslak, David A., T. Ryan Hoens, Nitesh V. Chawla, and W. Philip Kegelmeyer.
       "Hellinger distance decision trees are robust and skew-insensitive."
       Data Mining and Knowledge Discovery 24, no. 1 (2012): 136-158.

    """

    def __init__(self, min_branch_frac_option=0.01):
        super().__init__()
        self.min_branch_frac_option = min_branch_frac_option
        self.lowest_entropy = None
        self.best_idx = 0

    def get_merit_of_split(self, pre_split_dist, post_split_dist):
        if self.num_subsets_greater_than_frac(post_split_dist, self.min_branch_frac_option) < 2:
            return -np.inf
        return self.compute_hellinger(post_split_dist)

    @staticmethod
    def compute_hellinger(dist):

        try:
            left_branch_positive = dist[0][1]
            left_branch_negative = dist[0][0]
            right_branch_positive = dist[1][1]
            right_branch_negative = dist[1][0]
        except KeyError:
            return 0
        total_negative = left_branch_negative + right_branch_negative
        total_positive = left_branch_positive + right_branch_positive

        hellinger = (np.sqrt(left_branch_negative/total_negative)
                     - np.sqrt(left_branch_positive/total_positive)) ** 2 + \
                    (np.sqrt(right_branch_negative/total_negative)
                     - np.sqrt(right_branch_positive/total_positive)) ** 2

        return np.sqrt(hellinger)

    @staticmethod
    def get_range_of_merit(pre_split_dist):
        num_classes = len(pre_split_dist)
        num_classes = num_classes if num_classes > 2 else 2
        return np.log2(num_classes)

    @staticmethod
    def num_subsets_greater_than_frac(distributions, min_frac):
        total_weight = 0.0
        dist_sums = [0.0]*len(distributions)
        for i in range(len(dist_sums)):
            dist_sums[i] = sum(distributions[i].values())
            total_weight += dist_sums[i]
        num_greater = 0
        for d in dist_sums:
            if (d / total_weight) > min_frac:
                num_greater += 1
        return num_greater
