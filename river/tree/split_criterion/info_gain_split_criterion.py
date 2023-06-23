from __future__ import annotations

import math

from .base import SplitCriterion


class InfoGainSplitCriterion(SplitCriterion):
    """Information Gain split criterion.

    A measure of how often a randomly chosen element from the set would be
    incorrectly labeled if it was randomly labeled according to the
    distribution of labels in the subset.

    References
    ----------
    [Wikipedia entry](https://en.wikipedia.org/wiki/Decision_tree_learning#Information_gain)

    """

    def __init__(self, min_branch_fraction):
        super().__init__()
        self.min_branch_fraction = min_branch_fraction

    def merit_of_split(self, pre_split_dist, post_split_dist):
        if self.num_subsets_greater_than_frac(post_split_dist, self.min_branch_fraction) < 2:
            return -math.inf
        return self.compute_entropy(pre_split_dist) - self.compute_entropy(post_split_dist)

    @staticmethod
    def range_of_merit(pre_split_dist):
        num_classes = len(pre_split_dist)
        num_classes = num_classes if num_classes > 2 else 2
        return math.log2(num_classes)

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
            if d > 0.0:  # TODO: How small can d be before log2 overflows?
                entropy -= d * math.log2(d)
                dis_sums += d
        return (entropy + dis_sums * math.log2(dis_sums)) / dis_sums if dis_sums > 0.0 else 0.0

    def _compute_entropy_list(self, dists):
        total_weight = 0.0
        dist_weights = [0.0] * len(dists)
        for i in range(len(dists)):
            dist_weights[i] = sum(dists[i].values())
            total_weight += dist_weights[i]
        entropy = 0.0
        for i in range(len(dists)):
            entropy += dist_weights[i] * self.compute_entropy(dists[i])
        return entropy / total_weight

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
