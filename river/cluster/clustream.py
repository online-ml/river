import math
from abc import ABCMeta

from river import base, cluster, utils

EPSILON = 0.00005
MIN_VARIANCE = 1e-50


class CluStream(base.Clusterer):
    """CluStream

    The CluStream algorithm [^1] maintains statistical information about the
    data using micro-clusters. These micro-clusters are temporal extensions of
    cluster feature vectors. The micro-clusters are stored at snapshots in time
    following a pyramidal pattern. This pattern allows to recall summary
    statistics from different time horizons.

    Training with a new point `p` is performed in two main tasks:

    * Determinate closest micro-cluster to `p`

    * Check whether `p` fits (memory) into the closest micro-cluster:

        - if `p` fits, put into micro-cluster

        - if `p` does not fit, free some space to insert a new micro-cluster.
          This is done in two ways, delete an old micro-cluster or merge the
          two micro-clusters closest to each other.

    Parameters
    ----------
    seed
       Random seed used for generating initial centroid positions.

    time_window
        If the current time is `T` and the time window is `h`, we only consider
        the data that arrived within the period `(T-h,T)`.

    max_micro_clusters
        The maximum number of micro-clusters to use.

    micro_cluster_r_factor
        Multiplier for the micro-cluster radius.
        When deciding to add a new data point to a micro-cluster, the maximum
        boundary is defined as a factor of the `micro_cluster_r_factor` of
        the RMS deviation of the data points in the micro-cluster from the
        centroid.

    n_macro_clusters
        The number of clusters (k) for the k-means algorithm.

    kwargs
        Other parameters passed to the incremental kmeans at `cluster.KMeans`.

    Attributes
    ----------
    centers : dict
        Central positions of each cluster.

    References
    ----------
    [^1]: Aggarwal, C.C., Philip, S.Y., Han, J. and Wang, J., 2003,
          A framework for clustering evolving data streams.
          In Proceedings 2003 VLDB conference (pp. 81-92). Morgan Kaufmann.

    Examples
    --------

    In the following example, `max_micro_clusters` and `time_window` are set
    relatively low due to the limited number of training points.
    Moreover, all points are learnt before any predictions are made.
    The `halflife` is set at 0.4, to show that you can pass `cluster.KMeans`
    parameters via keyword arguments.

    >>> from river import cluster
    >>> from river import stream

    >>> X = [
    ...     [1, 2],
    ...     [1, 4],
    ...     [1, 0],
    ...     [4, 2],
    ...     [4, 4],
    ...     [4, 0]
    ... ]

    >>> clustream = cluster.CluStream(time_window=1,
    ...                               max_micro_clusters=3,
    ...                               n_macro_clusters=2,
    ...                               seed=0,
    ...                               halflife=0.4)

    >>> for i, (x, _) in enumerate(stream.iter_array(X)):
    ...     clustream = clustream.learn_one(x)

    >>> clustream.predict_one({0: 1, 1: 1})
    1

    >>> clustream.predict_one({0: 4, 1: 3})
    0

    """

    def __init__(
        self,
        seed: int = None,
        time_window: int = 1000,
        max_micro_clusters: int = 100,
        micro_cluster_r_factor: int = 2,
        n_macro_clusters: int = 5,
        **kwargs
    ):
        super().__init__()
        self.time_window = time_window
        self.time_stamp = -1
        self.micro_clusters = {n: None for n in range(max_micro_clusters)}
        self.initialized = False
        self.buffer = {}
        self.max_micro_clusters = max_micro_clusters
        self.centers = {}
        self.micro_cluster_r_factor = micro_cluster_r_factor
        self.max_micro_clusters = max_micro_clusters
        self.n_macro_clusters = n_macro_clusters
        self._train_weight_seen_by_model = 0.0
        self.seed = seed
        self.kwargs = kwargs

    def _initialize(self, x, sample_weight):

        # Create a micro-cluster with the new point
        if len(self.buffer) < self.max_micro_clusters:
            self.buffer[len(self.buffer)] = CluStreamMicroCluster(
                x=x,
                sample_weight=sample_weight,
                timestamp=self.time_stamp,
                micro_cluster_r_factor=self.micro_cluster_r_factor,
                max_micro_clusters=self.max_micro_clusters,
            )
        else:
            # The buffer is full. Use the micro-clusters centers to create the
            # micro-clusters set.
            for i in range(self.max_micro_clusters):
                self.micro_clusters[i] = CluStreamMicroCluster(
                    x=self.buffer[i].center,
                    sample_weight=1.0,
                    timestamp=self.time_stamp,
                    micro_cluster_r_factor=self.micro_cluster_r_factor,
                    max_micro_clusters=self.max_micro_clusters,
                )
            self.buffer.clear()
            self.initialized = True

    def _maintain_micro_clusters(self, x, sample_weight):
        # Calculate the threshold to delete old micro-clusters
        threshold = self.time_stamp - self.time_window

        # Delete old micro-clusters if its relevance stamp is smaller than the threshold
        for i, micro_cluster_a in self.micro_clusters.items():
            if micro_cluster_a.relevance_stamp < threshold:
                self.micro_clusters[i] = CluStreamMicroCluster(
                    x=x,
                    sample_weight=sample_weight,
                    timestamp=self.time_stamp,
                    micro_cluster_r_factor=self.micro_cluster_r_factor,
                    max_micro_clusters=self.max_micro_clusters,
                )
                return self

        # Merge the two closest micro-clusters
        closest_a = 0
        closest_b = 0
        min_distance = math.inf
        for i, micro_cluster_a in self.micro_clusters.items():
            for j, micro_cluster_b in self.micro_clusters.items():
                dist = self._distance(micro_cluster_a.center, micro_cluster_b.center)
                if dist < min_distance and j > i:
                    min_distance = dist
                    closest_a = i
                    closest_b = j
        self.micro_clusters[closest_a].add(self.micro_clusters[closest_b])
        self.micro_clusters[closest_b] = CluStreamMicroCluster(
            x=x,
            sample_weight=sample_weight,
            timestamp=self.time_stamp,
            micro_cluster_r_factor=self.micro_cluster_r_factor,
            max_micro_clusters=self.max_micro_clusters,
        )

    def _get_micro_clustering_result(self):
        if not self.initialized:
            return {}
        res = {
            i: CluStreamMicroCluster(
                micro_cluster=micro_cluster,
                micro_cluster_r_factor=self.micro_cluster_r_factor,
                max_micro_clusters=self.max_micro_clusters,
            )
            for i, micro_cluster in self.micro_clusters.items()
        }
        return res

    def _get_closest_micro_cluster(self, x, micro_clusters):
        min_distance = math.inf
        closest_micro_cluster_idx = -1
        for i, micro_cluster in micro_clusters.items():
            distance = self._distance(micro_cluster.center, x)
            if distance < min_distance:
                min_distance = distance
                closest_micro_cluster_idx = i
        return closest_micro_cluster_idx, min_distance

    @staticmethod
    def _distance(point_a, point_b):
        return math.sqrt(utils.math.minkowski_distance(point_a, point_b, 2))

    def learn_one(self, x, sample_weight=None):

        if sample_weight == 0:
            return
        elif sample_weight is None:
            sample_weight = 1.0

        self._train_weight_seen_by_model += sample_weight

        # merge _learn_one into learn_one
        self.time_stamp += 1

        if not self.initialized:
            self._initialize(x=x, sample_weight=sample_weight)
            return self

        # determine the closest micro-cluster with respect to the new point instance
        closest_micro_cluster = None
        min_distance = math.inf
        for micro_cluster in self.micro_clusters.values():
            distance = self._distance(x, micro_cluster.center)
            if distance < min_distance:
                closest_micro_cluster = micro_cluster
                min_distance = distance

        # check whether the new instance fits into the closest micro-cluster
        if closest_micro_cluster.weight == 1:
            radius = math.inf
            center = closest_micro_cluster.center
            for micro_cluster in self.micro_clusters.values():
                if micro_cluster == closest_micro_cluster:
                    continue
                distance = self._distance(micro_cluster.center, center)
                radius = min(distance, radius)
        else:
            radius = closest_micro_cluster.radius

        if min_distance < radius:
            closest_micro_cluster.insert(x, sample_weight, self.time_stamp)
            return self

        # If the new point does not fit in the micro-cluster, micro-clusters
        # whose relevance stamps are less than the threshold are deleted.
        # Otherwise, closest micro-clusters are merged with each other.
        self._maintain_micro_clusters(x=x, sample_weight=sample_weight)

        return self

    def predict_one(self, x):

        micro_cluster_centers = {
            i: self._get_micro_clustering_result()[i].center
            for i in range(len(self._get_micro_clustering_result()))
        }

        kmeans = cluster.KMeans(
            n_clusters=self.n_macro_clusters, seed=self.seed, **self.kwargs
        )
        for center in micro_cluster_centers.values():
            kmeans = kmeans.learn_one(center)

        self.centers = kmeans.centers

        index, _ = self._get_closest_micro_cluster(
            x, self._get_micro_clustering_result()
        )
        y = kmeans.predict_one(micro_cluster_centers[index])

        return y


class CluStreamMicroCluster(metaclass=ABCMeta):
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

    def add(self, micro_cluster):
        self.n_samples += micro_cluster.n_samples
        self.linear_sum_timestamp += micro_cluster.linear_sum_timestamp
        self.squared_sum_timestamp += micro_cluster.squared_sum_timestamp
        utils.skmultiflow_utils.add_dict_values(
            self.linear_sum, micro_cluster.linear_sum, inplace=True
        )
        utils.skmultiflow_utils.add_dict_values(
            self.squared_sum, micro_cluster.squared_sum, inplace=True
        )
