from river import base
import math
import sys
from river.cluster.denstream.denstream_microcluster import MicroCluster
from sklearn.cluster import DBSCAN
import numpy as np


class Denstream(base.Clusterer):
    def __init__(self, decaying_factor=None, core_weight_threshold=None, tolerance_factor=None):
        super().__init__()
        self.time_stamp = -1
        self.buffer = {} # buffer for outliers
        self.l = decaying_factor
        self.mu = core_weight_threshold
        self.beta = tolerance_factor

    def _T_p(self):
        return math.ceil(1 / self.l * math.log(self.beta * self.mu / (self.beta * self.mu - 1)))

    def add_closest_cluster(self, new_point, micro_cluster, timestamp):
        min_distance = sys.float_info.max
        closest_cluster_index = -1
        for i, micro_cluster in micro_cluster.items():
            distance = self._distance(micro_cluster.get_center(), new_point)
            if distance < min_distance:
                min_distance = distance
                closest_kernel_index = i
        micro_cluster[closest_cluster_index].add(new_point, timestamp)

    def _distance(v1, v2):
        distance = 0.0
        for i in range(len(v1)):
            d = v1[i] - v2[i]
            distance += d * d
        return math.sqrt(distance)