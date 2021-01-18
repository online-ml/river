import collections

from river import base
from abc import ABCMeta, abstractmethod

EPSILON = 0.00005
MIN_VARIANCE = 1e-50

# from file clustream_kernel.py in scikit-multiflow/clustering

class ClustreamKernel(base.Clusterer, metaclass = ABCMeta):

    def __init__(self, x=None, weight=None, cluster=None, dimensions=None, timestamp=None, T=None, M=None):
        super().__init__()

        self.T = T
        self.M = M
        # check if the new instance has the same length
        if x is None and weight is None and cluster is None and dimensions is not None:
            self.N = 0
            self.LS = {n: 0.0 for n in range(len(x))}
            self.SS = {n: 0.0 for n in range(dimensions)}
        elif x is not None and weight is not None and dimensions is not None:
            super().__init__(x=x, weight=weight, dimensions=dimensions)
            self.N = 1
            self.LS = {}
            self.SS = {}
            for i in range(len(x)):
                self.LS[i] = x[i] * weight
                self.SS[i] = x[i] * x[i] * weight
            self.LST = timestamp * weight
            self.SST = timestamp * timestamp * weight
        elif cluster is not None:
            super().__init__(cluster=cluster)
            self.N = cluster.N
            self.LS = cluster.LS.copy()
            self.SS = cluster.SS.copy()
            self.LST = cluster.LST
            self.SST = cluster.SST

    def get_center(self):
        assert not self.is_empty()
        res = {n: 0.0 for n in range(len(self.LS))}
        for i in range(len(res)):
            res[i] = self.LS[i] / self.N
        return res

    def is_empty(self):
        return self.N == 0

    def get_radius(self):
        if self.N == 1:
            return 0
        return self.get_deviation() * self.T

    def get_deviation(self):
        variance = self.get_variance_vector()
        sum_of_deviation = 0
        for i in range(len(variance)):
            d = math.sqrt(variance[i])
            sum_of_deviation += d
        return sum_of_deviation / len(variance)

    def get_variance_vector(self):
        res = {n: 0.0 for n in range(len(self.LS))}
        for i in range(len(self.LS)):
            ls = self.LS[i]
            ss = self.SS[i]
            ls_div_n = ls / self.get_weight()
            ls_div_n_squared = ls_div_n * ls_div_n
            ss_div_n = ss / self.get_weight()
            res[i] = ss_div_n - ls_div_n_squared

            if res[i] <= 0.0:
                if res[i] > - EPSILON:
                    res[i] = MIN_VARIANCE
        return res

    # implemented from cluster_feature.py
    def get_weight(self):
        return self.N

    def insert(self, x, weight, timestamp):
        self.N += weight
        self.LST += timestamp * weight
        self.SST += timestamp * weight
        for i in range(len(x)):
            self.LS[i] += x[i] * weight
            self.SS[i] += x[i] * x[i] * weight

    def get_relevance_stamp(self):
        if self.N < 2 * self.M:
            return self.get_mu_time()
        return self.get_mu_time() + self.get_sigma_time() * self.get_quantile(float(self.M)/(2 * self.N))

    def get_mu_time(self):
        return self.LST / self.N

    def get_sigma_time(self):
        return math.sqrt(self.SST/self.N - (self.LST/self.N) * (self.LST/self.N))

    def get_quantile(self, z):
        assert (z >= 0 and z <= 1)
        return math.sqrt(2) * self.inverse_error(2 * z - 1)

    @staticmethod
    def inverse_error(x):
        z = math.sqrt(math.pi) * x
        res = x / 2
        z2 = z * z

        zprod = z2 * z
        res += (1.0 / 24) * zprod

        zprod *= z2  # z5
        res += (7.0 / 960) * zprod

        zprod *= z2  # z ^ 7
        res += (127 * zprod) / 80640

        zprod *= z2  # z ^ 9
        res += (4369 * zprod) / 11612160

        zprod *= z2  # z ^ 11
        res += (34807 * zprod) / 364953600

        zprod *= z2  # z ^ 13
        res += (20036983 * zprod) / 797058662400

        return res

    def add(self, cluster):
        assert len(cluster.LS) == len(self.LS)
        self.N += cluster.N
        self.LST += cluster.LST
        self.SST += cluster.SST
        self.add_vectors(self.LS, cluster.LS)
        self.add_vectors(self.SS, cluster.SS)

    # implemented from cluster_feature.py
    def add_vectors(v1, v2):
        assert v1 is not None
        assert v2 is not None
        assert len(v1) == len(v2)
        for i in range(len(v1)):
            v1[i] += v2[i]

    @abstractmethod
    def get_CF(self):
        return self

    def sample(self, random_state):
        raise NotImplementedError

