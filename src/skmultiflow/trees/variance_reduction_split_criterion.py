from skmultiflow.trees.split_criterion import SplitCriterion
import numpy as np


class VarianceReductionSplitCriterion(SplitCriterion):

    def __init__(self):
        super().__init__()

    def get_merit_of_split(self, pre_split_dist, post_split_dist):
        SDR = 0.0
        N = pre_split_dist[0]

        count = 0
        for i in range(len(post_split_dist)):
            Ni = post_split_dist[i][0]
            if Ni >= 5.0:
                count += 1
        if count == len(post_split_dist):

            SDR = self.compute_SD(pre_split_dist)
            for i in range(len(post_split_dist)):
                Ni = post_split_dist[i][0]
                SDR -= Ni/N * self.compute_SD(post_split_dist[i])
        return SDR

    @staticmethod
    def compute_SD(dist):

        N = int(dist[0])
        sum = dist[1]
        sum_sq = dist[2]

        return np.sqrt((sum_sq - (sum * sum)/N)/N)

    @staticmethod
    def get_range_of_merit(pre_split_dist):
        return 1.0
