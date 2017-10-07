__author__ = 'Jacob Montiel'

from skmultiflow.classification.core.split_criteria.split_criterion import SplitCriterion

class GiniSplitCriterion(SplitCriterion):
    def get_merit_of_split(self, pre_split_dist, post_split_dist):
        total_weight = 0.0
        dist_weights = []*len(post_split_dist)
        for i in range(len(post_split_dist)):
            dist_weights[i] = sum(post_split_dist[i].values())
            total_weight += dist_weights[i]
        gini = 0.0
        for i in range(len(post_split_dist)):
            gini += (dist_weights[i] / total_weight) * self.compute_gini(post_split_dist[i], dist_weights[i])
        return 1.0 - gini

    def compute_gini(self, dist, dist_sum_of_weights):
        gini = 1.0
        for _, val in dist.items():
            rel_freq = val / dist_sum_of_weights
            gini -= rel_freq * rel_freq
        return gini

    def get_range_of_merit(self, pre_split_dist):
        return 1.0