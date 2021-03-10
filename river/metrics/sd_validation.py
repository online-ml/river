import math

from river import stats, utils

from . import base_internal_clustering

__all__ = ["SD"]


class SD(base_internal_clustering.InternalClusteringMetrics):
    """The SD validity index (SD).

    The SD validity index (SD) is a more recent clustering validation measure. It is composed of
    two terms:

    * Scat(NC) stands for the scattering within clusters,

    * Dis(NC) stands for the dispersion between clusters.

    Like DB and SB, SD measures the compactness with variance of clustered objects and separation
    with distnace between cluster centers, but uses them in a different way. The smaller the value
    of SD, the better.

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
    >>> metric = metrics.SD()

    >>> for x, _ in stream.iter_array(X):
    ...     k_means = k_means.learn_one(x)
    ...     y_pred = k_means.predict_one(x)
    ...     metric = metric.update(k_means.centers, x, y_pred)

    >>> metric
    SD: 4.332702

    """

    def __init__(self):
        super().__init__()
        self._center_all_points = {}
        self._overall_variance = {}
        self._cluster_variance = {}
        self._centers = {}
        self._initialized = False

    @staticmethod
    def _calculate_dispersion_nc(centers):
        min_distance_clusters = math.inf
        max_distance_clusters = -math.inf
        sum_inverse_distances = 0

        n_clusters = len(centers)

        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                distance_ij = math.sqrt(
                    utils.math.minkowski_distance(centers[i], centers[j], 2)
                )
                if distance_ij > max_distance_clusters:
                    max_distance_clusters = distance_ij
                if distance_ij < min_distance_clusters:
                    min_distance_clusters = distance_ij
                sum_inverse_distances += 1 / distance_ij

        return max_distance_clusters / min_distance_clusters * sum_inverse_distances

    @staticmethod
    def _norm(x):
        origin = {i: 0 for i in x}
        return math.sqrt(utils.math.minkowski_distance(x, origin, 2))

    def update(self, centers, point, y_pred, sample_weight=1.0):

        if not self._initialized:
            self._center_all_points = self._overall_variance = {
                i: stats.Mean() for i in point
            }
            self._initialized = True
        for i in self._center_all_points:
            self._center_all_points[i].update(point[i], w=sample_weight)
        center_all_points = {
            i: self._center_all_points[i].get() for i in self._center_all_points
        }

        for i in self._overall_variance:
            self._overall_variance[i].update(
                (point[i] - center_all_points[i]) ** 2, w=sample_weight
            )

        if y_pred not in self._cluster_variance:
            self._cluster_variance[y_pred] = {i: stats.Mean() for i in point}
        for i in point:
            self._cluster_variance[y_pred][i].update(
                (point[i] - centers[y_pred][i]) ** 2, w=sample_weight
            )

        self._centers = centers

        return self

    def revert(self, centers, point, y_pred, sample_weight=1.0, correction=None):

        for i in self._center_all_points:
            self._center_all_points[i].update(point[i], w=-sample_weight)
        center_all_points = {
            i: self._center_all_points[i].get() for i in self._center_all_points
        }

        for i in self._overall_variance:
            self._overall_variance[i].update(
                (point[i] - center_all_points[i]) ** 2, w=-sample_weight
            )

        for i in point:
            self._cluster_variance[y_pred][i].update(
                (point[i] - centers[y_pred][i]) ** 2, w=-sample_weight
            )

        self._centers = centers

        return self

    def get(self):

        dispersion_nc = self._calculate_dispersion_nc(self._centers)

        overall_variance = {
            i: self._overall_variance[i].get() for i in self._overall_variance
        }
        cluster_variance = {}
        for i in self._cluster_variance:
            cluster_variance[i] = {
                j: self._cluster_variance[i][j].get() for j in self._cluster_variance[i]
            }

        scat_nc = 0
        for i in cluster_variance:
            scat_nc += self._norm(cluster_variance[i]) / self._norm(overall_variance)

        return scat_nc + dispersion_nc

    @property
    def bigger_is_better(self):
        return False
