from __future__ import annotations

import math

from .base import SplitCriterion


class GiniSplitCriterion(SplitCriterion):
    """Gini Impurity split criterion."""

    def __init__(self, min_branch_fraction):
        super().__init__()
        self.min_branch_fraction = min_branch_fraction

    def merit_of_split(self, pre_split_dist, post_split_dist):
        if self.num_subsets_greater_than_frac(post_split_dist, self.min_branch_fraction) < 2:
            return -math.inf

        total_weight = 0.0
        dist_weights = [0.0] * len(post_split_dist)
        for i in range(len(post_split_dist)):
            dist_weights[i] = sum(post_split_dist[i].values())
            total_weight += dist_weights[i]
        gini = 0.0
        for i in range(len(post_split_dist)):
            gini += (dist_weights[i] / total_weight) * self.compute_gini(
                post_split_dist[i], dist_weights[i]
            )
        return 1.0 - gini

    @staticmethod
    def compute_gini(dist, dist_sum_of_weights):
        gini = 1.0
        if dist_sum_of_weights != 0.0:
            for _, val in dist.items():
                rel_freq = val / dist_sum_of_weights
                gini -= rel_freq * rel_freq
        return gini

    @staticmethod
    def range_of_merit(pre_split_dist):
        return 1.0

    @staticmethod
    def num_subsets_greater_than_frac(distributions, min_frac):
        total_weight = 0.0
        dist_sums = [0.0] * len(distributions)
        for i in range(len(dist_sums)):
            dist_sums[i] = sum(distributions[i].values())
            total_weight += dist_sums[i]
        num_greater = 0

        if total_weight > 0:
            for d in dist_sums:
                if (d / total_weight) > min_frac:
                    num_greater += 1
        return num_greater
