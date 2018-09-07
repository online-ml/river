from skmultiflow.trees.variance_reduction_split_criterion \
    import VarianceReductionSplitCriterion
import numpy as np


# This class extends VarianceReductionSplitCriterion since it just computes
# the SD differently than its Ancestor (considering multiple targets)
class IntraClusterVarianceReductionSplitCriterion(VarianceReductionSplitCriterion):

    def __init__(self):
        super().__init__()

    @staticmethod
    def compute_SD(dist):
        # TODO Also consider passing different weights for the targets

        N = int(dist[0])
        sum = dist[1]
        sum_sq = dist[2]

        return np.sum(
            (sum_sq - (sum ** 2) / N) / (N - 1)
        )
