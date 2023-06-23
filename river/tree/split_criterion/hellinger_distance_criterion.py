from __future__ import annotations

import math

from .base import SplitCriterion


class HellingerDistanceCriterion(SplitCriterion):
    """Hellinger Distance split criterion.

    The Hellinger distance is a measure of distributional divergence.
    It is used as the splitting criterion [^1] on decision trees to to address
    the imbalanced data problem.

    References
    ----------
    [^1]: Cieslak, David A., T. Ryan Hoens, Nitesh V. Chawla, and W. Philip Kegelmeyer.
    "Hellinger distance decision trees are robust and skew-insensitive." Data Mining
    and Knowledge Discovery 24, no. 1 (2012): 136-158.
    """

    def __init__(self, min_branch_fraction):
        super().__init__()
        self.min_branch_fraction = min_branch_fraction

    def merit_of_split(self, pre_split_dist, post_split_dist):
        if self.num_subsets_greater_than_frac(post_split_dist, self.min_branch_fraction) < 2:
            return -math.inf
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

        hellinger = (
            math.sqrt(left_branch_negative / total_negative)
            - math.sqrt(left_branch_positive / total_positive)
        ) ** 2 + (
            math.sqrt(right_branch_negative / total_negative)
            - math.sqrt(right_branch_positive / total_positive)
        ) ** 2

        return math.sqrt(hellinger)

    @staticmethod
    def range_of_merit(pre_split_dist):
        num_classes = len(pre_split_dist)
        num_classes = num_classes if num_classes > 2 else 2
        return math.log2(num_classes)

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
