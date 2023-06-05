from __future__ import annotations

import math
from collections import defaultdict

from river import base, cluster, stats, utils


class CluStream(base.Clusterer):
    """CluStream

    The CluStream algorithm [^1] maintains statistical information about the
    data using micro-clusters. These micro-clusters are temporal extensions of
    cluster feature vectors. The micro-clusters are stored at snapshots in time
    following a pyramidal pattern. This pattern allows to recall summary
    statistics from different time horizons.

    Training with a new point `p` is performed in two main tasks:

    * Determinate the closest micro-cluster to `p`.

    * Check whether `p` fits (memory) into the closest micro-cluster:

        - if `p` fits, put into micro-cluster

        - if `p` does not fit, free some space to insert a new micro-cluster.
          This is done in two ways, delete an old micro-cluster or merge the
          two micro-clusters closest to each other.

    This implementation is an improved version from the original algorithm. Instead
    of calculating the traditional cluster feature vector of the number of observations,
    linear sum and sum of squares of data points and time stamps, this implementation adopts
    the use of Welford's algorithm [^2] to calculate the incremental variance, facilitated
    through `stats.Var` available within River.

    Since River does not support an actual "off-line" phase of the clustering algorithm (as data
    points are assumed to arrive continuously, one at a time), a `time_gap` parameter is introduced.
    After each `time_gap`, an incremental K-Means clustering algorithm will be initialized and
    applied on currently available micro-clusters to form the final solution, i.e. macro-clusters.

    Parameters
    ----------
    n_macro_clusters
        The number of clusters (k) for the k-means algorithm.
    max_micro_clusters
        The maximum number of micro-clusters to use.
    micro_cluster_r_factor
        Multiplier for the micro-cluster radius. When deciding to add a new data point to a micro-cluster,
        the maximum boundary is defined as a factor of the `micro_cluster_r_factor` of
        the RMS deviation of the data points in the micro-cluster from the centroid.
    time_window
        If the current time is `T` and the time window is `h`, we only consider
        the data that arrived within the period `(T-h,T)`.
    time_gap
        An incremental k-means is applied on the current set of micro-clusters after each `time_gap` to form
        the final macro-cluster solution.
    seed
       Random seed used for generating initial centroid positions.
    kwargs
        Other parameters passed to the incremental kmeans at `cluster.KMeans`.

    Attributes
    ----------
    centers : dict
        Central positions of each cluster.

    References
    ----------
    [^1]: Aggarwal, C.C., Philip, S.Y., Han, J. and Wang, J., 2003, A framework for clustering evolving data
    streams. In Proceedings 2003 VLDB conference (pp. 81-92). Morgan Kaufmann.
    [^2]: Chan, T.F., Golub, G.H. and LeVeque, R.J., 1982. Updating formulae and a pairwise algorithm for
    computing sample variances. In COMPSTAT 1982 5th Symposium held at Toulouse 1982 (pp. 30-41).
    Physica, Heidelberg. https://doi.org/10.1007/978-3-642-51461-6_3.

    Examples
    --------

    In the following example, `max_micro_clusters` is set relatively low due to the
    limited number of training points. Moreover, all points are learnt before any predictions are made.
    The `halflife` is set at 0.4, to show that you can pass `cluster.KMeans` parameters via keyword arguments.

    >>> from river import cluster
    >>> from river import stream

    >>> X = [
    ...     [1, 2],
    ...     [1, 4],
    ...     [1, 0],
    ...     [-4, 2],
    ...     [-4, 4],
    ...     [-4, 0],
    ...     [5, 0],
    ...     [5, 2],
    ...     [5, 4]
    ... ]

    >>> clustream = cluster.CluStream(
    ...     n_macro_clusters=3,
    ...     max_micro_clusters=5,
    ...     time_gap=3,
    ...     seed=0,
    ...     halflife=0.4
    ... )

    >>> for x, _ in stream.iter_array(X):
    ...     clustream = clustream.learn_one(x)

    >>> clustream.predict_one({0: 1, 1: 1})
    1

    >>> clustream.predict_one({0: -4, 1: 3})
    2

    >>> clustream.predict_one({0: 4, 1: 3.5})
    0

    """

    def __init__(
        self,
        n_macro_clusters: int = 5,
        max_micro_clusters: int = 100,
        micro_cluster_r_factor: int = 2,
        time_window: int = 1000,
        time_gap: int = 100,
        seed: int | None = None,
        **kwargs,
    ):
        super().__init__()
        self.n_macro_clusters = n_macro_clusters
        self.max_micro_clusters = max_micro_clusters
        self.micro_cluster_r_factor = micro_cluster_r_factor
        self.time_window = time_window
        self.time_gap = time_gap
        self.seed = seed

        self.kwargs = kwargs

        self.centers: dict[int, defaultdict] = {}
        self.micro_clusters: dict[int, CluStreamMicroCluster] = {}

        self._timestamp = -1
        self._initialized = False

        self._mc_centers: dict[int, defaultdict] = {}
        self._kmeans_mc = None

    def _maintain_micro_clusters(self, x, w):
        # Calculate the threshold to delete old micro-clusters
        threshold = self._timestamp - self.time_window

        # Delete old micro-cluster if its relevance stamp is smaller than the threshold
        del_id = None
        for i, mc in self.micro_clusters.items():
            if mc.relevance_stamp(self.max_micro_clusters) < threshold:
                del_id = i
                break

        if del_id is not None:
            self.micro_clusters[del_id] = CluStreamMicroCluster(
                x=x,
                w=w,
                timestamp=self._timestamp,
            )
            return

        # Merge the two closest micro-clusters
        closest_a = 0
        closest_b = 0
        min_distance = math.inf
        for i, mc_a in self.micro_clusters.items():
            for j, mc_b in self.micro_clusters.items():
                if i <= j:
                    continue
                dist = self._distance(mc_a.center, mc_b.center)
                if dist < min_distance:
                    min_distance = dist
                    closest_a = i
                    closest_b = j

        self.micro_clusters[closest_a] += self.micro_clusters[closest_b]
        self.micro_clusters[closest_b] = CluStreamMicroCluster(
            x=x,
            w=w,
            timestamp=self._timestamp,
        )

    def _get_closest_mc(self, x):
        closest_dist = math.inf
        closest_idx = -1

        for mc_idx, mc in self.micro_clusters.items():
            distance = self._distance(mc.center, x)
            if distance < closest_dist:
                closest_dist = distance
                closest_idx = mc_idx
        return closest_idx, closest_dist

    @staticmethod
    def _distance(point_a, point_b):
        return utils.math.minkowski_distance(point_a, point_b, 2)

    def learn_one(self, x, w=1.0):
        self._timestamp += 1

        if not self._initialized:
            self.micro_clusters[len(self.micro_clusters)] = CluStreamMicroCluster(
                x=x,
                w=w,
                # When initialized, all micro clusters generated previously will have the timestamp reset to the current
                # time stamp at the time of initialization (i.e. self.max_micro_cluster - 1). Thus, the timestamp is set
                # as follows.
                timestamp=self.max_micro_clusters - 1,
            )

            if len(self.micro_clusters) == self.max_micro_clusters:
                self._initialized = True

            return self

        # Determine the closest micro-cluster with respect to the new point instance
        closest_id, closest_dist = self._get_closest_mc(x)
        closest_mc = self.micro_clusters[closest_id]

        # Check whether the new instance fits into the closest micro-cluster
        if closest_mc.weight == 1:
            radius = math.inf
            center = closest_mc.center
            for mc_id, mc in self.micro_clusters.items():
                if mc_id == closest_id:
                    continue
                distance = self._distance(mc.center, center)
                radius = min(distance, radius)
        else:
            radius = closest_mc.radius(self.micro_cluster_r_factor)

        if closest_dist < radius:
            closest_mc.insert(x, w, self._timestamp)
            return self

        # If the new point does not fit in the micro-cluster, micro-clusters
        # whose relevance stamps are less than the threshold are deleted.
        # Otherwise, closest micro-clusters are merged with each other.
        self._maintain_micro_clusters(x=x, w=w)

        # Apply incremental K-Means on micro-clusters after each time_gap
        if self._timestamp % self.time_gap == self.time_gap - 1:
            # Micro-cluster centers will only be saved when the calculation of macro-cluster centers
            # is required, in order not to take up memory and time unnecessarily
            self._mc_centers = {i: mc.center for i, mc in self.micro_clusters.items()}

            self._kmeans_mc = cluster.KMeans(
                n_clusters=self.n_macro_clusters, seed=self.seed, **self.kwargs
            )
            for center in self._mc_centers.values():
                self._kmeans_mc = self._kmeans_mc.learn_one(center)

            self.centers = self._kmeans_mc.centers

        return self

    def predict_one(self, x):
        index, _ = self._get_closest_mc(x)
        try:
            return self._kmeans_mc.predict_one(self._mc_centers[index])
        except (KeyError, AttributeError):
            return 0


class CluStreamMicroCluster(base.Base):
    """Micro-cluster class."""

    def __init__(
        self,
        x: dict = defaultdict(float),
        w: float | None = None,
        timestamp: int | None = None,
    ):
        # Initialize with sample x
        self.x = x
        self.w = w
        self.timestamp = timestamp
        self.var_x = {k: stats.Var().update(x[k], w) for k in x}
        self.var_time = stats.Var().update(timestamp, w)

    @property
    def center(self):
        return {k: var.mean.get() for k, var in self.var_x.items()}

    def radius(self, r_factor):
        if self.weight == 1:
            return 0
        return self._deviation() * r_factor

    def _deviation(self):
        dev_sum = 0
        for var in self.var_x.values():
            dev_sum += math.sqrt(var.get())
        return dev_sum / len(self.var_x) if len(self.var_x) > 0 else math.inf

    @property
    def weight(self):
        return self.var_time.n

    def insert(self, x, w, timestamp):
        self.var_time.update(timestamp, w)
        for x_idx, x_val in x.items():
            self.var_x[x_idx].update(x_val, w)

    def relevance_stamp(self, max_mc):
        mu_time = self.var_time.mean.get()
        if self.weight < 2 * max_mc:
            return mu_time

        sigma_time = math.sqrt(self.var_time.get())
        return mu_time + sigma_time * self._quantile(max_mc / (2 * self.weight))

    def _quantile(self, z):
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

    def __iadd__(self, other: CluStreamMicroCluster):
        self.var_time += other.var_time
        self.var_x = {k: self.var_x[k] + other.var_x.get(k, stats.Var()) for k in self.var_x}
        return self
