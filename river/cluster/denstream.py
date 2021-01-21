from river import base
from abc import ABCMeta, abstractmethod
import math
import sys
from river.cluster.denstream_microcluster import MicroCluster
import numpy as np


class Denstream(base.Clusterer, metaclass=ABCMeta):
    def __init__(self, decaying_factor=None, core_weight_threshold=None, tolerance_factor=None, eps=None):
        super().__init__()
        self.time_stamp = -1
        self.o_micro_clusters = {}
        self.p_micro_clusters = {}
        self.l = decaying_factor
        self.mu = core_weight_threshold
        self.beta = tolerance_factor
        self.eps = eps

    def _T_p(self):
        return math.ceil(1 / self.l * math.log(self.beta * self.mu / (self.beta * self.mu - 1)))

    def find_closest_cluster_index(self, point, micro_clusters):
        min_distance = sys.float_info.max
        closest_cluster_index = -1
        for i, micro_cluster in micro_clusters.items():
            distance = self.get_distance(micro_clusters.get_center(), point)
            if distance < min_distance:
                min_distance = distance
                closest_cluster_index = i
        return closest_cluster_index

    def merge(self, point):
        # try to merge p into nearest c_p
        closest_p_micro_cluster_index = self.find_closest_cluster_index(point, self.p_micro_clusters, self.time_stamp)
        new_p_micro_cluster = self.p_micro_clusters[closest_p_micro_cluster_index].add(point, self.time_stamp)
        if new_p_micro_cluster.get_radius() <= self.eps:
            # merge p into nearest c_p
            self.p_micro_clusters[closest_p_micro_cluster_index].add(point, self.time_stamp)
        else:
            # try to merge p into nearest c_0
            closest_o_micro_cluster_index = self.find_closest_cluster_index(point, self.o_micro_clusters, self.time_stamp)
            new_o_micro_cluster = self.o_micro_clusters[closest_o_micro_cluster_index].add(point, self.time_stamp)
            if new_o_micro_cluster.get_radius <= self.eps:
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

    @staticmethod
    def get_distance(v1, v2):
        distance = 0.0
        for i in range(len(v1)):
            d = v1[i] - v2[i]
            distance += d * d
        return math.sqrt(distance)

    def learn_one(self, x, sample_weight=None):

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
                xi = (2 ** (- self.l * (self.time_stamp - o_micro_cluster.t_0 + self._T_p())) - 1)  / (2 ** (self.l * self._T_p()) - 1)
                if o_micro_cluster.get_weight() < xi:
                    # delete c_o
                    self.o_micro_clusters.pop(j)

    def is_directly_density_reachable(self, c_p, c_q):
        if c_p.get_weight(self.time_stamp) > self.mu and c_q.get_weight(self.time_stamp) > self.mu:
            if self.get_distance(c_p.get_center(), c_q.get_center()) < 2 * self.eps:
                if self.get_distance(c_p.get_center(), c_q.get_center()) < c_p.get_radius() + c_q.get_radius():
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False

    def find_neighbors(self, cluster):
        neighbors = {}
        for p_micro_cluster in self.p_micro_clusters.values():
            if self.is_directly_density_reachable(cluster, p_micro_cluster) and cluster != p_micro_cluster:
                neighbors[p_micro_cluster] = p_micro_cluster
        return neighbors

    def generate_cluster(self):
        # variance of the original DBSCAN
        # initiate dictionary of labels of the type cluster: label
        labels = {val: val for val in self.p_micro_clusters.values()}
        # cluster counter
        C = 0
        for p_micro_cluster in self.p_micro_clusters.values():
            neighbors = self.find_neighbors(p_micro_cluster)
            C += 1
            # label(P) = C
            labels[p_micro_cluster] = C

            for neighbor in neighbors.values():
                if labels[neighbor] == neighbor:
                    neighbor_neighbors = self.find_neighbors(neighbor)
                    for neighbor_neighbor in neighbor_neighbors.keys():
                        neighbors[neighbor_neighbor] = neighbor_neighbor
