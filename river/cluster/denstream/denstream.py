from river import base
import math
import sys
from river.cluster.denstream.denstream_microcluster import MicroCluster
from sklearn.cluster import DBSCAN
import numpy as np


class Denstream(base.Clusterer):
    def __init__(self, decaying_factor=None, core_weight_threshold=None, tolerance_factor=None, esp=None):
        super().__init__()
        self.time_stamp = -1
        self.o_micro_clusters = {}
        self.p_micro_clusters = {}
        self.l = decaying_factor
        self.mu = core_weight_threshold
        self.beta = tolerance_factor
        self.esp = esp

    def _T_p(self):
        return math.ceil(1 / self.l * math.log(self.beta * self.mu / (self.beta * self.mu - 1)))

    def find_closest_cluster_index(self, point, micro_clusters):
        min_distance = sys.float_info.max
        closest_cluster_index = -1
        for i, micro_cluster in micro_clusters.items():
            distance = self._distance(micro_clusters.get_center(), point)
            if distance < min_distance:
                min_distance = distance
                closest_cluster_index = i
        return closest_cluster_index

    def merge(self, point):
        # try to merge p into nearest c_p
        closest_p_micro_cluster_index = self.find_closest_cluster_index(point, self.p_micro_clusters, self.time_stamp)
        new_p_micro_cluster = self.p_micro_clusters[closest_p_micro_cluster_index].add(point, self.time_stamp)
        if new_p_micro_cluster.get_radius() <= self.esp:
            # merge p into nearest c_p
            self.p_micro_clusters[closest_p_micro_cluster_index].add(point, self.time_stamp)
        else:
            # try to merge p into nearest c_0
            closest_o_micro_cluster_index = self.find_closest_cluster_index(point, self.o_micro_clusters, self.time_stamp)
            new_o_micro_cluster = self.o_micro_clusters[closest_o_micro_cluster_index].add(point, self.time_stamp)
            if new_o_micro_cluster.get_radius <= self.esp:
                # merge p into nearest c_0
                self.o_micro_clusters[closest_o_micro_cluster_index].add(point, self.time_stamp)
                if self.o_micro_clusters[closest_o_micro_cluster_index].get_weight() > self.beta * self.mu:
                    # remove c_0 from outlier-buffer
                    self.o_micro_clusters.pop(closest_o_micro_cluster_index)
                    # add a new p_micro_cluster by c_o
                    self.p_micro_clusters[len(self.p_micro_clusters) + 1] = new_o_micro_cluster
            else:
                # create a new o-micro-cluster by p {0: point} and add it to o_micro_clusters
                self.o_micro_clusters[len(self.o_micro_clusters) + 1] = {0: point}

    def _distance(v1, v2):
        distance = 0.0
        for i in range(len(v1)):
            d = v1[i] - v2[i]
            distance += d * d
        return math.sqrt(distance)

    def learn_one(self, x):

        self.time_stamp += 1

        # merging p
        self.merge(self, x)

        # run through if conditions
        if self.time_stamp % self._T_p() == 0:
            for i, p_micro_cluster in self.p_micro_clusters:
                if p_micro_cluster.get_weight() < self.beta * self.mu:
                    # delete c_p
                    self.p_micro_clusters.pop(i)
            for j, o_micro_cluster in self.o_micro_clusters:
                xi = (2 ** ( - self.l * (self.time_stamp - o_micro_cluster.t_0 + self._T_p())) - 1)  / (2 ** (self.l * self._T_p()) - 1)
                if o_micro_cluster.get_weight() < xi:
                    # delete c_o
                    self.o_micro_clusters.pop(j)

    def generate_cluster(self):
        #TODO




