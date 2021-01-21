from abc import ABCMeta
import math

EPSILON = 0.00005
MIN_VARIANCE = 1e-50

# from file clustream_kernel.py in scikit-multiflow/clustering


class ClustreamKernel(metaclass=ABCMeta):

    def __init__(self, x=None, sample_weight=None, cluster=None, timestamp=None, T=None, M=None):
        # super().__init__()

        self.T = T
        self.M = M

        # check if the new instance has the same length
        # remove the case that x is None, because it would be impossible to get len(x)
        if x is not None and sample_weight is not None:
            # super().__init__(x=x, sample_weight=sample_weight)
            self.N = 1
            self.LS = {}
            self.SS = {}
            for i in range(len(x)):
                self.LS[i] = x[i] * sample_weight
                self.SS[i] = x[i] * x[i] * sample_weight
            self.LST = timestamp * sample_weight
            self.SST = timestamp * timestamp * sample_weight
        elif cluster is not None:
            # super().__init__(cluster=cluster)
            self.N = cluster.N
            self.LS = cluster.LS.copy()
            self.SS = cluster.SS.copy()
            self.LST = cluster.LST
            self.SST = cluster.SST

    @property
    def center(self):
        res = {n: 0.0 for n in range(len(self.LS))}
        for i in range(len(res)):
            res[i] = self.LS[i] / self.N
        return res

    def is_empty(self):
        return self.N == 0

    @property
    def radius(self):
        if self.N == 1:
            return 0
        return self._deviation * self.T

    @property
    def _deviation(self):
        variance = self._variance_vector
        sum_of_deviation = 0
        for i in range(len(variance)):
            d = math.sqrt(variance[i])
            sum_of_deviation += d
        return sum_of_deviation / len(variance)

    @property
    def _variance_vector(self):
        res = {}
        for i in range(len(self.LS)):
            ls = self.LS[i]
            ss = self.SS[i]
            ls_div_n = ls / self.weight
            ls_div_n_squared = ls_div_n * ls_div_n
            ss_div_n = ss / self.weight
            res[i] = ss_div_n - ls_div_n_squared

            if res[i] <= 0.0:
                if res[i] > - EPSILON:
                    res[i] = MIN_VARIANCE
        return res

    @property
    def weight(self):
        return self.N

    def insert(self, x, sample_weight, timestamp):
        self.N += sample_weight
        self.LST += timestamp * sample_weight
        self.SST += timestamp * sample_weight
        for i in range(len(x)):
            self.LS[i] += x[i] * sample_weight
            self.SS[i] += x[i] * x[i] * sample_weight

    @property
    def relevance_stamp(self):
        if self.N < 2 * self.M:
            return self._mu_time
        return self._mu_time + self._sigma_time * self._quantile(float(self.M)/(2 * self.N))

    @property
    def _mu_time(self):
        return self.LST / self.N

    @property
    def _sigma_time(self):
        return math.sqrt(self.SST/self.N - (self.LST/self.N) * (self.LST/self.N))

    def _quantile(self, z):
        assert 0 <= z <= 1
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

    @staticmethod
    def add_vectors(v1, v2):
        assert v1 is not None
        assert v2 is not None
        assert len(v1) == len(v2)
        for i in range(len(v1)):
            v1[i] += v2[i]
