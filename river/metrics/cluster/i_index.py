import math

from river import stats, utils

from . import base


class IIndex(base.InternalMetric):
    """I-Index (I).

    I-Index (I) [^1] adopts the maximum distance between cluster centers. It also shares the type of
    formulation numerator-separation/denominator-compactness. For compactness, the distance from
    a data point to its cluster center is also used like CH.

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
    >>> metric = metrics.cluster.IIndex()

    >>> for x, _ in stream.iter_array(X):
    ...     k_means = k_means.learn_one(x)
    ...     y_pred = k_means.predict_one(x)
    ...     metric = metric.update(x, y_pred, k_means.centers)

    >>> metric
    IIndex: 6.836566

    References
    --------

    [^1]: Maulik, U., Bandyopadhyay, S. (2002). Performance evaluation of some clustering algorithms and
          validity indices. In: IEEE Transactions on Pattern Analysis and Machine Intelligence 24(12)
          1650 - 1654. DOI: 10.1109/TPAMI.2002.1114856

    """

    def __init__(self):
        super().__init__()
        self._center_all_points = {}
        self._ssq_points_cluster_centers = 0
        self._ssq_points_center = 0
        self._furthest_cluster_distance = 0
        self._n_clusters = 0
        self._dim = 0
        self.sample_correction = {}
        self._initialized = False

    @staticmethod
    def _find_furthest_cluster_distance(centers):
        n_centers = len(centers)
        max_distance = -math.inf
        for i in range(n_centers):
            for j in range(i + 1, n_centers):
                distance_ij = math.sqrt(
                    utils.math.minkowski_distance(centers[i], centers[j], 2)
                )
                if distance_ij > max_distance:
                    max_distance = distance_ij
        return max_distance

    def update(self, x, y_pred, centers, sample_weight=1.0):

        self._furthest_cluster_distance = self._find_furthest_cluster_distance(centers)

        if not self._initialized:
            self._center_all_points = {i: stats.Mean() for i in x}
            self._dim = len(x)
            self._initialized = True

        for i in self._center_all_points:
            self._center_all_points[i].update(x[i], w=sample_weight)
        center_all_points = {
            i: self._center_all_points[i].get() for i in self._center_all_points
        }

        distance_point_cluster_center = math.sqrt(
            utils.math.minkowski_distance(centers[y_pred], x, 2)
        )
        distance_point_center = math.sqrt(
            utils.math.minkowski_distance(center_all_points, x, 2)
        )
        self._ssq_points_cluster_centers += distance_point_cluster_center
        self._ssq_points_center += distance_point_center
        self._n_clusters = len(centers)

        # To trace back
        self.sample_correction = {
            "distance_point_cluster_center": distance_point_cluster_center,
            "distance_point_center": distance_point_center,
        }

        return self

    def revert(self, x, y_pred, centers, sample_weight=1.0, correction=None):

        self._furthest_cluster_distance = self._find_furthest_cluster_distance(centers)

        for i in self._center_all_points:
            self._center_all_points[i].update(x[i], w=-sample_weight)

        self._ssq_points_cluster_centers -= correction["distance_point_cluster_center"]
        self._ssq_points_center -= correction["distance_point_center"]
        self._n_clusters = len(centers)
        self._dim = len(x)

        return self

    def get(self):
        try:
            return (
                1
                / self._n_clusters
                * self._ssq_points_center
                / self._ssq_points_cluster_centers
                * self._furthest_cluster_distance
            ) ** self._dim
        except ZeroDivisionError:
            return -math.inf

    @property
    def bigger_is_better(self):
        return True
