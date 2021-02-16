import math
from abc import ABCMeta

EPSILON = 0.00005
MIN_VARIANCE = 1e-50


class CluStreamKernel(metaclass=ABCMeta):
    def __init__(
        self,
        x=None,
        sample_weight=None,
        cluster=None,
        timestamp=None,
        kernel_radius_factor=None,
        max_kernels=None,
    ):

        self.kernel_radius_factor = kernel_radius_factor
        self.max_kernels = max_kernels

        # check if the new instance has the same length
        # remove the case that x is None, because it would be impossible to get len(x)
        if x is not None and sample_weight is not None:
            self.n_samples = 1
            self.linear_sum = {}
            self.squared_sum = {}
            for i in range(len(x)):
                self.linear_sum[i] = x[i] * sample_weight
                self.squared_sum[i] = x[i] * x[i] * sample_weight
            self.linear_sum_timestamp = timestamp * sample_weight
            self.squared_sum_timestamp = timestamp * timestamp * sample_weight
        elif cluster is not None:
            self.n_samples = cluster.n_samples
            self.linear_sum = cluster.linear_sum.copy()
            self.squared_sum = cluster.squared_sum.copy()
            self.linear_sum_timestamp = cluster.linear_sum_timestamp
            self.squared_sum_timestamp = cluster.squared_sum_timestamp

    @property
    def center(self):
        res = {
            i: self.linear_sum[i] / self.n_samples for i in range(len(self.linear_sum))
        }
        return res

    def is_empty(self):
        return self.n_samples == 0

    @property
    def radius(self):
        if self.n_samples == 1:
            return 0
        return self._deviation * self.kernel_radius_factor

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
        for i in range(len(self.linear_sum)):
            ls = self.linear_sum[i]
            ss = self.squared_sum[i]
            ls_div_n = ls / self.weight
            ls_div_n_squared = ls_div_n * ls_div_n
            ss_div_n = ss / self.weight
            res[i] = ss_div_n - ls_div_n_squared

            if res[i] <= 0.0:
                if res[i] > -EPSILON:
                    res[i] = MIN_VARIANCE
        return res

    @property
    def weight(self):
        return self.n_samples

    def insert(self, x, sample_weight, timestamp):
        self.n_samples += sample_weight
        self.linear_sum_timestamp += timestamp * sample_weight
        self.squared_sum_timestamp += timestamp * sample_weight
        for i in range(len(x)):
            self.linear_sum[i] += x[i] * sample_weight
            self.squared_sum[i] += x[i] * x[i] * sample_weight

    @property
    def relevance_stamp(self):
        if self.n_samples < 2 * self.max_kernels:
            return self._mu_time
        return self._mu_time + self._sigma_time * self._quantile(
            float(self.max_kernels) / (2 * self.n_samples)
        )

    @property
    def _mu_time(self):
        return self.linear_sum_timestamp / self.n_samples

    @property
    def _sigma_time(self):
        return math.sqrt(
            self.squared_sum_timestamp / self.n_samples
            - (self.linear_sum_timestamp / self.n_samples)
            * (self.linear_sum_timestamp / self.n_samples)
        )

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
        if len(cluster.linear_sum) == len(self.linear_sum):
            self.n_samples += cluster.n_samples
            self.linear_sum_timestamp += cluster.linear_sum_timestamp
            self.squared_sum_timestamp += cluster.squared_sum_timestamp
            self.add_vectors(self.linear_sum, cluster.linear_sum)
            self.add_vectors(self.squared_sum, cluster.squared_sum)

    @staticmethod
    def add_vectors(v1, v2):
        assert v1 is not None
        assert v2 is not None
        assert len(v1) == len(v2)
        for i in range(len(v1)):
            v1[i] += v2[i]
