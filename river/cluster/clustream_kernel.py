import math
from abc import ABCMeta

from river.utils.skmultiflow_utils import add_dict_values

EPSILON = 0.00005
MIN_VARIANCE = 1e-50


class MicroCluster(metaclass=ABCMeta):
    """ Micro-cluster class """

    def __init__(
        self,
        x: dict = None,
        sample_weight: float = None,
        micro_cluster=None,
        timestamp: int = None,
        micro_cluster_r_factor: int = None,
        max_micro_clusters: int = None,
    ):

        self.micro_cluster_r_factor = micro_cluster_r_factor
        self.max_micro_clusters = max_micro_clusters

        if x is not None and sample_weight is not None:
            # Initialize with sample x
            self.n_samples = 1
            self.linear_sum = {}
            self.squared_sum = {}
            for key in x.keys():
                self.linear_sum[key] = x[key] * sample_weight
                self.squared_sum[key] = x[key] * x[key] * sample_weight
            self.linear_sum_timestamp = timestamp * sample_weight
            self.squared_sum_timestamp = timestamp * timestamp * sample_weight
        elif micro_cluster is not None:
            # Initialize with micro-cluster
            self.n_samples = micro_cluster.n_samples
            self.linear_sum = micro_cluster.linear_sum.copy()
            self.squared_sum = micro_cluster.squared_sum.copy()
            self.linear_sum_timestamp = micro_cluster.linear_sum_timestamp
            self.squared_sum_timestamp = micro_cluster.squared_sum_timestamp

    @property
    def center(self):
        return {
            i: linear_sum_i / self.n_samples
            for i, linear_sum_i in self.linear_sum.items()
        }

    def is_empty(self):
        return self.n_samples == 0

    @property
    def radius(self):
        if self.n_samples == 1:
            return 0
        return self._deviation * self.micro_cluster_r_factor

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
        for key in self.linear_sum.keys():
            ls = self.linear_sum[key]
            ss = self.squared_sum[key]
            ls_div_n = ls / self.weight
            ls_div_n_squared = ls_div_n * ls_div_n
            ss_div_n = ss / self.weight
            res[key] = ss_div_n - ls_div_n_squared

            if res[key] <= 0.0:
                if res[key] > -EPSILON:
                    res[key] = MIN_VARIANCE
        return res

    @property
    def weight(self):
        return self.n_samples

    def insert(self, x, sample_weight, timestamp):
        self.n_samples += 1
        self.linear_sum_timestamp += timestamp * sample_weight
        self.squared_sum_timestamp += timestamp * sample_weight
        for i in range(len(x)):
            self.linear_sum[i] += x[i] * sample_weight
            self.squared_sum[i] += x[i] * x[i] * sample_weight

    @property
    def relevance_stamp(self):
        if self.n_samples < 2 * self.max_micro_clusters:
            return self._mu_time
        return self._mu_time + self._sigma_time * self._quantile(
            float(self.max_micro_clusters) / (2 * self.n_samples)
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
            add_dict_values(self.linear_sum, cluster.linear_sum, inplace=True)
            add_dict_values(self.squared_sum, cluster.squared_sum, inplace=True)
