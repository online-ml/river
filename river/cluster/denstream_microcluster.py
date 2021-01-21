import collections

from river import base
from abc import ABCMeta, abstractmethod
import math


class MicroCluster(base.Clusterer, metaclass = ABCMeta):
    def __init__(self, p=None, timestamps=None, creation_time=None, decaying_factor=None):
        # p: dictionary of dictionaries (points p_1, ..., p_n)
        # timestamps: dictionary of n items (timestamps of points T_1, T_2, ..., T_n)
        # creation_time: creation time t_0 of the cluster

        super().__init__()

        self.p = p
        self.timestamps = timestamps
        self.t_0 = creation_time
        self.l = decaying_factor

        if p is not None and timestamps is not None and decaying_factor is not None:
            assert len(p) == len(timestamps)
            assert decaying_factor > 0
            self.n = len(p) # number of points
            for i in range(self.n):
                for j in range(i, self.n):
                    assert len(p[i]) == len(p[j])  # check all points have the same dimensions
            self.dim = len(p[0])  # dimension of the points

    def f(self, t):
        return 2 ** (-self.l * t)

    # t_current: current time t for calculating parameters

    def get_weight(self, t_current):
        weight = 0
        for i in range(self.n):
            weight += self.f(t_current - self.timestamps[i])
        return weight

    def get_weighted_linear_sum(self, t_current):
        weighted_linear_sum = {i: 0.0 for i in range(self.dim)}
        for i in range(self.n):
            for j in range(self.dim):
                weighted_linear_sum[j] += self.f(t_current - self.timestamps[i]) * self.p[i][j]
        return weighted_linear_sum

    def get_weighted_square_sum(self, t_current):
        weighted_square_sum = {i: 0.0 for i in range(self.dim)}
        for i in range(self.n):
            for j in range(self.dim):
                weighted_square_sum[j] += self.f(t_current - self.timestamps[i]) * self.p[i][j]
        return weighted_square_sum

    def get_center(self):
        center = {i: 0.0 for i in range(self.dim)}
        for i in range(self.dim):
            center[i] = self.get_weighted_linear_sum()[i] / self.get_weight()
        return center

    @staticmethod
    def get_distance(v1, v2):
        distance = 0.0
        for i in range(len(v1)):
            d = v1[i] - v2[i]
            distance += d * d
        return math.sqrt(distance)

    def get_radius(self, t_current):
        radius = 0
        weight = self.get_weight()
        center = self.get_center()
        for i in range(self.n):
            radius += self.f(t_current - self.timestamps[i]) * self.get_distance(self.p[i], center) / weight
        return radius

    def add(self, new_point, t_current):
        self.n += 1
        self.p[self.n] = new_point
        self.timestamps[self.n] = t_current