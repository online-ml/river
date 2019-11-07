from skmultiflow.trees.split_criterion import VarianceReductionSplitCriterion
import numpy as np


# This class extends VarianceReductionSplitCriterion since it just computes
# the SD differently than its Ancestor (considering multiple targets)
class IntraClusterVarianceReductionSplitCriterion(
        VarianceReductionSplitCriterion):

    def __init__(self):
        super().__init__()

    def get_merit_of_split(self, pre_split_dist, post_split_dist):
        SDR = 0.0
        N = pre_split_dist[0]

        count = 0

        for dist in post_split_dist:
            Ni = dist[0]
            if Ni >= 5.0:
                count += 1

        if count == len(post_split_dist):
            SDR = self.compute_SD(pre_split_dist)
            for dist in post_split_dist:
                Ni = dist[0]
                SDR -= Ni/N * self.compute_SD(dist)
        return SDR

    @staticmethod
    def compute_SD(dist):
        # TODO Also consider passing different weights for the targets

        N = dist[0]
        sum = dist[1]
        sum_sq = dist[2]

        return np.mean((sum_sq - ((sum * sum) / N)) / (N - 1))
