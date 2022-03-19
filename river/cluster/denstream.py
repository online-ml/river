import copy
import math
from abc import ABCMeta
from collections import defaultdict, deque

from river import base, utils


class DenStream(base.Clusterer):
    r"""DenStream

    DenStream [^1] is a clustering algorithm for evolving data streams.
    DenStream can discover clusters with arbitrary shape and is robust against
    noise (outliers).

    "Dense" micro-clusters (named core-micro-clusters) summarise the clusters
    of arbitrary shape. A pruning strategy based on the concepts of potential
    and outlier micro-clusters guarantees the precision of the weights of the
    micro-clusters with limited memory.

    The algorithm is divided into two parts:

    **Online micro-cluster maintenance (learning)**

    For a new point `p`:

    * Try to merge `p` into either the nearest `p-micro-cluster` (potential),
    `o-micro-cluster` (outlier), or create a new `o-micro-cluster` and insert it
    into the outlier buffer.

    * For each `T_p` iterations, consider the weights of all potential and
    outlier micro-clusters. If their weights are smaller than a certain
    threshold (different for each type of micro-clusters), the micro-cluster is
    deleted.

    **Offline generation of clusters on-demand (clustering)**

    A variant of the DBSCAN algorithm [^2] is used, such that all
    density-connected p-micro-clusters determine the final clusters.

    Parameters
    ----------
    decaying_factor
        Parameter that controls the importance of historical data to current cluster.
        Note that `decaying_factor` has to be different from `0`.

    beta
        Parameter to determine the threshold of outlier relative to core micro-clusters.
        Valid values are `0 < \beta <= 1`.

    mu
        Parameter to determine the threshold of outliers relative to core micro-cluster.
        Valid values are `\mu > 0`.

    epsilon
        Defines the epsilon neighborhood

    n_samples_init
        Number of points to to initiqalize the online process

    stream_speed
        Number of points arrived in unit time

    Attributes
    ----------
    n_clusters
        Number of clusters generated by the algorithm.

    clusters
        A set of final clusters of type `MicroCluster`, which means that these cluster include all
        the required information, including number of points, creation time, weight, (weighted)
        linear sum, (weighted) square sum, center and radius.

    p_micro_clusters
        The potential core-icro-clusters that are generated by the algorithm. When a generate
        cluster request arrives, these p-micro-clusters will go through a variant of the DBSCAN
        algorithm to determine the final clusters.

    o_micro_clusters
        The outlier micro-clusters.

    References
    ----------
    [^1]: Feng et al (2006, pp 328-339). Density-Based Clustering over an Evolving Data Stream with
          Noise. In Proceedings of the Sixth SIAM International Conference on Data Mining,
          April 20–22, 2006, Bethesda, MD, USA.
    [^2]: Ester et al (1996). A Density-Based Algorithm for Discovering Clusters in Large Spatial
          Databases with Noise. In KDD-96 Proceedings, AAAI.

    Examples
    ----------

    The following example uses the default parameters of the algorithm to test its functionality.
    The set of evolving points `X` are designed so that clusters are easily identifiable.

    >>> from river import cluster
    >>> from river import stream

    >>> X = [
    ...     [-1, -0.5], [-1, -0.625], [-1, -0.75], [-1, -1], [-1, -1.125],
    ...     [-1, -1.25], [-1.5, -0.5], [-1.5, -0.625], [-1.5, -0.75], [-1.5, -1],
    ...     [-1.5, -1.125], [-1.5, -1.25], [1, 1.5], [1, 1.75], [1, 2],
    ...     [4, 1.25], [4, 1.5], [4, 2.25], [4, 2.5], [4, 3],
    ...     [4, 3.25], [4, 3.5], [4, 3.75], [4, 4],
    ... ]

    >>> denstream = cluster.DenStream(decaying_factor = 0.01,
    ...                               beta = 0.5,
    ...                               mu = 2.5,
    ...                               epsilon = 0.5,
    ...                               n_samples_init=10)

    >>> for x, _ in stream.iter_array(X):
    ...     denstream = denstream.learn_one(x)

    >>> denstream.predict_one({0: -1, 1: -2})
    1

    >>> denstream.predict_one({0:5, 1:4})
    0

    >>> denstream.predict_one({0:1, 1:1})
    1

    >>> denstream.n_clusters
    2

    """

    class BufferItem:
        def __init__(self, x, timestamp, covered):
            self.x = x
            self.timestamp = (timestamp,)
            self.covered = covered

    def __init__(
        self,
        decaying_factor: float = 0.25,
        beta: float = 5,
        mu: float = 0.5,
        epsilon: float = 0.02,
        n_samples_init: int = 1000,
        stream_speed: int = 100,
    ):
        super().__init__()
        self.timestamp = -1
        self.initialized = False
        self.decaying_factor = decaying_factor
        self.beta = beta
        self.mu = mu
        self.epsilon = epsilon
        self.n_samples_init = n_samples_init
        self.stream_speed = stream_speed

        # number of clusters generated by applying the variant of DBSCAN algorithm
        # on p-micro-cluster centers and their centers
        self.n_clusters = 0
        self.clusters = {}
        self.p_micro_clusters = {}
        self.o_micro_clusters = {}

        self._time_period = math.ceil(
            (1 / self.decaying_factor)
            * math.log((self.mu * self.beta) / (self.mu * self.beta - 1))
        )
        self._init_buffer = deque()
        self._n_samples_seen = 0

        # check that the value of beta is within the range (0,1]
        if not (0 < self.beta <= 1):
            raise ValueError(f"The value of `beta` must be within the range (0,1].")

    @property
    def centers(self):
        return {
            k: cluster.calc_center(self.timestamp)
            for k, cluster in self.clusters.items()
        }

    @staticmethod
    def _distance(point_a, point_b):
        return math.sqrt(utils.math.minkowski_distance(point_a, point_b, 2))

    def _get_closest_cluster_key(self, point, clusters):
        min_distance = math.inf
        key = -1
        for k, cluster in clusters.items():
            center = cluster.calc_center(self.timestamp)
            distance = self._distance(center, point)
            if distance < min_distance:
                min_distance = distance
                key = k
        return key

    def _merge(self, point):
        # initiate merged status
        merged_status = False

        if len(self.p_micro_clusters) != 0:
            # try to merge p into its nearest p-micro-cluster c_p
            closest_pmc_key = self._get_closest_cluster_key(
                point, self.p_micro_clusters
            )
            updated_pmc = copy.copy(self.p_micro_clusters[closest_pmc_key])
            updated_pmc.insert(point, self.timestamp)
            if updated_pmc.calc_radius(self.timestamp) <= self.epsilon:
                # keep updated p-micro-cluster
                self.p_micro_clusters[closest_pmc_key] = updated_pmc
                merged_status = True

        if not merged_status and len(self.o_micro_clusters) != 0:
            closest_omc_key = self._get_closest_cluster_key(
                point, self.o_micro_clusters
            )
            updated_omc = copy.copy(self.o_micro_clusters[closest_omc_key])
            updated_omc.insert(point, self.timestamp)
            if updated_omc.calc_radius(self.timestamp) <= self.epsilon:
                # keep updated o-micro-cluster
                if updated_omc.calc_weight(self.timestamp) > self.mu * self.beta:
                    # it has grown into a p-micro-cluster
                    del self.o_micro_clusters[closest_omc_key]
                    self.p_micro_clusters[len(self.p_micro_clusters)] = updated_omc
                else:
                    self.o_micro_clusters[closest_omc_key] = updated_omc
            else:
                # create a new o-micro-cluster by p and add it to o_micro_clusters
                mc_from_p = DenStreamMicroCluster(
                    x=point,
                    timestamp=self.timestamp,
                    decaying_factor=self.decaying_factor,
                )
                self.o_micro_clusters[len(self.o_micro_clusters)] = mc_from_p

    def _is_directly_density_reachable(self, c_p, c_q):
        if (
            c_p.calc_weight(self.timestamp) > self.mu
            and c_q.calc_weight(self.timestamp) > self.mu
        ):
            # check distance of two clusters and compare with 2*epsilon
            c_p_center = c_p.calc_center(self.timestamp)
            c_q_center = c_q.calc_center(self.timestamp)
            distance = self._distance(c_p_center, c_q_center)
            if distance < 2 * self.epsilon and distance <= c_p.calc_radius(
                self.timestamp
            ) + c_q.calc_radius(self.timestamp):
                return True
        return False

    def _query_neighbor(self, cluster):
        neighbors = deque()
        # scan all clusters within self.p_micro_clusters
        for pmc in self.p_micro_clusters.values():
            # check density reachable and that the cluster itself does not appear in neighbors
            if cluster != pmc and self._is_directly_density_reachable(cluster, pmc):
                neighbors.append(pmc)
        return neighbors

    @staticmethod
    def _generate_clusters_for_labels(cluster_labels):
        # initiate the dictionary for final clusters
        clusters = {}

        # group clusters per label
        mcs_per_label = defaultdict(deque)
        for mc, label in cluster_labels.items():
            mcs_per_label[label].append(mc)

        # generate set of clusters with the same label
        for label, micro_clusters in mcs_per_label.items():
            # merge clusters with the same label into a big cluster
            cluster = copy.copy(micro_clusters[0])
            for mc in range(1, len(micro_clusters)):
                cluster.merge(micro_clusters[mc])

            clusters[label] = cluster

        return len(clusters), clusters

    def _expand_cluster(self, mc, neighborhood):
        for idx in neighborhood:
            item = self._init_buffer[idx]
            if not item.covered:
                item.covered = True
                mc.insert(item.x, self.timestamp)

    def _get_neighborhood_ids(self, item):
        neighborhood_ids = deque()
        for idx, other in enumerate(self._init_buffer):
            if not other.covered:
                if self._distance(item.x, other.x) < self.epsilon:
                    neighborhood_ids.append(idx)
        return neighborhood_ids

    def _initial_dbscan(self):
        for item in self._init_buffer:
            if not item.covered:
                item.covered = True
                neighborhood = self._get_neighborhood_ids(item)
                if len(neighborhood) > self.mu:
                    mc = DenStreamMicroCluster(
                        x=item.x,
                        timestamp=self.timestamp,
                        decaying_factor=self.decaying_factor,
                    )
                    self._expand_cluster(mc, neighborhood)
                    self.p_micro_clusters.update({len(self.p_micro_clusters): mc})
                else:
                    item.covered = False

    def learn_one(self, x, sample_weight=None):
        self._n_samples_seen += 1
        # control the stream speed
        if self._n_samples_seen % self.stream_speed == 0:
            self.timestamp += 1

        # Initialization
        if not self.initialized:
            self._init_buffer.append(self.BufferItem(x, self.timestamp, False))
            if len(self._init_buffer) == self.n_samples_init:
                self._initial_dbscan()
                self.initialized = True
                del self._init_buffer
            return self

        # Merge
        self._merge(x)

        # Periodic cluster removal
        if self.timestamp > 0 and self.timestamp % self._time_period == 0:
            for i, p_micro_cluster_i in list(self.p_micro_clusters.items()):
                if p_micro_cluster_i.calc_weight(self.timestamp) < self.mu * self.beta:
                    # c_p became an outlier and should be deleted
                    del self.p_micro_clusters[i]
            for j, o_micro_cluster_j in list(self.o_micro_clusters.items()):
                # calculate xi
                xi = (
                    2
                    ** (
                        -self.decaying_factor
                        * (
                            self.timestamp
                            - o_micro_cluster_j.creation_time
                            + self._time_period
                        )
                    )
                    - 1
                ) / (2 ** (-self.decaying_factor * self._time_period) - 1)
                if o_micro_cluster_j.calc_weight(self.timestamp) < xi:
                    # c_o might not grow into a p-micro-cluster, we can safely delete it
                    self.o_micro_clusters.pop(j)
        return self

    def predict_one(self, x, sample_weight=None):

        # This function handles the case when a clustering request arrives.
        # implementation of the DBSCAN algorithm proposed by Ester et al.
        if not self.initialized:
            # The model is not ready
            return 0

        # cluster counter; in this algorithm cluster labels start with 0
        c = -1
        # initiate labels of p-micro-clusters to None
        labels = {pmc: None for pmc in self.p_micro_clusters.values()}

        for pmc in self.p_micro_clusters.values():
            # previously processed in inner loop
            if labels[pmc] is not None:
                continue
            # next cluster label
            c += 1
            labels[pmc] = c
            # neighbors to expand
            seed_queue = self._query_neighbor(pmc)
            # process every point in seed set
            while seed_queue:
                # check previously proceeded points
                if labels[seed_queue[0]] is not None:
                    seed_queue.popleft()
                    continue
                if seed_queue:
                    labels[seed_queue[0]] = c
                    # find neighbors of neighbors
                    neighbor_neighbors = self._query_neighbor(seed_queue[0])
                    # add new neighbors to seed set
                    for neighbor_neighbor in neighbor_neighbors:
                        if labels[neighbor_neighbor] is not None:
                            seed_queue.append(neighbor_neighbor)

        self.n_clusters, self.clusters = self._generate_clusters_for_labels(labels)

        return self._get_closest_cluster_key(x, self.clusters)


class DenStreamMicroCluster(metaclass=ABCMeta):
    """ DenStream Micro-cluster class """

    def __init__(self, x, timestamp, decaying_factor):

        self.x = x
        self.last_edit_time = timestamp
        self.creation_time = timestamp
        self.decaying_factor = decaying_factor

        self.N = 1
        self.linear_sum = x
        self.squared_sum = {i: (x_val * x_val) for i, x_val in x.items()}

    def calc_norm_cf1_cf2(self, fading_factor):
        # |CF1| and |CF2| in the paper
        sum_of_squares_cf1 = 0
        sum_of_squares_cf2 = 0
        for key in self.linear_sum.keys():
            val_ls = self.linear_sum[key]
            val_ss = self.squared_sum[key]
            sum_of_squares_cf1 += fading_factor * val_ls * fading_factor * val_ls
            sum_of_squares_cf2 += fading_factor * val_ss * fading_factor * val_ss
        # return |CF1| and |CF2|
        return math.sqrt(sum_of_squares_cf1), math.sqrt(sum_of_squares_cf2)

    def calc_weight(self, timestamp):
        return self._weight(self.fading_function(timestamp - self.last_edit_time))

    def _weight(self, fading_factor):
        return self.N * fading_factor

    def calc_center(self, timestamp):
        ff = self.fading_function(timestamp - self.last_edit_time)
        weight = self._weight(ff)
        center = {key: (ff * val) / weight for key, val in self.linear_sum.items()}
        return center

    def calc_radius(self, timestamp):
        ff = self.fading_function(timestamp - self.last_edit_time)
        weight = self._weight(ff)
        norm_cf1, norm_cf2 = self.calc_norm_cf1_cf2(ff)
        diff = (norm_cf2 / weight) - ((norm_cf1 / weight) ** 2)
        radius = math.sqrt(diff) if diff > 0 else 0
        return radius

    def insert(self, x, timestamp):
        self.N += 1
        self.last_edit_time = timestamp
        for key, val in x.items():
            try:
                self.linear_sum[key] += val
                self.squared_sum[key] += val * val
            except KeyError:
                self.linear_sum[key] = val
                self.squared_sum[key] = val * val

    def merge(self, cluster):
        self.N += cluster.N
        for key in cluster.linear_sum.keys():
            try:
                self.linear_sum[key] += cluster.linear_sum[key]
                self.squared_sum[key] += cluster.squared_sum[key]
            except KeyError:
                self.linear_sum[key] = cluster.linear_sum[key]
                self.squared_sum[key] = cluster.squared_sum[key]
        if self.last_edit_time < cluster.creation_time:
            self.last_edit_time = cluster.creation_time

    def fading_function(self, time):
        return 2 ** (-self.decaying_factor * time)
