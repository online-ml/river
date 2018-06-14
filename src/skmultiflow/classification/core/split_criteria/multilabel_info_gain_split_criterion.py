from skmultiflow.classification.core.split_criteria.info_gain_split_criterion import InfoGainSplitCriterion
import numpy as np


class MutltiLabelInfoGainSplitCriterion(InfoGainSplitCriterion):
    """docstring for MutltiLabelInfoGainSplitCriterion"""

    def __init__(self):
        super().__init__()

    def computer_entropy(self, dist):
        if isinstance(dist, dict):
            return self.compute_entropy_dict(dist)
        elif isinstance(dist, list):
            return self.compute_entropy_list(dist)

    def compute_entropy_dict(self, dist):
        entropy = 0.0
        dis_sums = 0.0
        for _, d in dist.items():
            if d > 0.0:
                entropy -= (d * np.log2(d) + (1 - d) * np.log2(1 - d))
                dis_sums += d
        return (entropy + dis_sums * np.log2(dis_sums)) / dis_sums if dis_sums > 0.0 else 0.0

    def compute_entropy_list(self, dist):
        total_weight = 0.0
        dist_weights = [0.0] * len(dist)
        for i in range(len(dist)):
            dist_weights[i] = sum(dist[i].values())
            total_weight += dist_weights[i]
        entropy = 0.0
        for i in range(len(dist)):
            entropy += dist_weights[i] * self.compute_entropy(dist[i])
        return entropy / total_weight
