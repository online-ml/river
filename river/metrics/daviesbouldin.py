import math

from river import utils

from . import base_internal_clustering

__all__ = ["DaviesBouldin"]


class DaviesBouldin(base_internal_clustering.InternalClusteringMetrics):
    """Davies-Bouldin index (DB).

    The Davies-Bouldin index (DB) [^1] is an old but still widely used inernal validaion measure.
    DB uses intra-cluster variance and inter-cluster center disance to find the worst partner
    cluster, i.e., the closest most scattered one for each cluster. Thus, minimizing DB gives
    us the optimal number of clusters.

    Examples
    --------

    >>> from river import cluster
    >>> from river import stream
    >>> from river import metrics

    >>> X = [
    ...     [1, 2],
    ...     [1, 4],
    ...     [1, 0],
    ...     [4, 2],
    ...     [4, 4],
    ...     [4, 0],
    ...     [-2, 2],
    ...     [-2, 4],
    ...     [-2, 0]
    ... ]

    >>> k_means = cluster.KMeans(n_clusters=3, halflife=0.4, sigma=3, seed=0)
    >>> metric = metrics.DaviesBouldin()

    >>> for x, _ in stream.iter_array(X):
    ...     k_means = k_means.learn_one(x)
    ...     y_pred = k_means.predict_one(x)
    ...     metric = metric.update(x, y_pred, k_means.centers)

    >>> metric
    DaviesBouldin: 0.22583

    References
    ----------
    [^1]: David L., D., Don, B. (1979). A Cluster Separation Measure. In: IEEE
          Transactions on Pattern Analysis and Machine Intelligence (PAMI) 1(2), 224 - 227.
          DOI: 10.1109/TPAMI.1979.4766909

    """

    def __init__(self):
        super().__init__()
        self._inter_cluster_distances = {}
        self._n_points_by_clusters = {}
        self._total_points = 0
        self._centers = {}
        self.sample_correction = {}

    def update(self, x, y_pred, centers, sample_weight=1.0):

        distance = math.sqrt(utils.math.minkowski_distance(centers[y_pred], x, 2))

        # To trace back
        self.sample_correction = {"distance": distance}

        if y_pred not in self._inter_cluster_distances:
            self._inter_cluster_distances[y_pred] = distance
            self._n_points_by_clusters[y_pred] = 1
        else:
            self._inter_cluster_distances[y_pred] += distance
            self._n_points_by_clusters[y_pred] += 1

        self._centers = centers

        return self

    def revert(self, x, y_pred, centers, sample_weight=1.0, correction=None):

        self._inter_cluster_distances[y_pred] -= correction["distance"]
        self._n_points_by_clusters[y_pred] -= 1
        self._centers = centers

        return self

    def get(self):
        max_partner_clusters_index = -math.inf
        n_clusters = len(self._inter_cluster_distances)
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                distance_ij = math.sqrt(
                    utils.math.minkowski_distance(self._centers[i], self._centers[j], 2)
                )
                ij_partner_cluster_index = (
                    self._inter_cluster_distances[i] / self._n_points_by_clusters[i]
                    + self._inter_cluster_distances[j] / self._n_points_by_clusters[j]
                ) / distance_ij
                if ij_partner_cluster_index > max_partner_clusters_index:
                    max_partner_clusters_index = ij_partner_cluster_index
        return max_partner_clusters_index / n_clusters

    @property
    def bigger_is_better(self):
        return False
