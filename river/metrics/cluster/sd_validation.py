import math

from river import stats, utils

from . import base


class SD(base.InternalMetric):
    """The SD validity index (SD).

    The SD validity index (SD) [^1] is a more recent clustering validation measure. It is composed of
    two terms:

    * Scat(NC) stands for the scattering within clusters,

    * Dis(NC) stands for the dispersion between clusters.

    Like DB and SB, SD measures the compactness with variance of clustered objects and separation
    with distance between cluster centers, but uses them in a different way. The smaller the value
    of SD, the better.

    In the original formula for SD validation index, the ratio between the maximum and the actual
    number of clusters is taken into account. However, due to the fact that metrics are updated in
    an incremental fashion, this ratio will be automatically set to default as 1.

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
    >>> metric = metrics.cluster.SD()

    >>> for x, _ in stream.iter_array(X):
    ...     k_means = k_means.learn_one(x)
    ...     y_pred = k_means.predict_one(x)
    ...     metric = metric.update(x, y_pred, k_means.centers)

    >>> metric
    SD: 2.339016

    References
    ----------
    [^1]: Halkidi, M., Vazirgiannis, M., & Batistakis, Y. (2000). Quality Scheme Assessment in the
          Clustering Process. Principles Of Data Mining And Knowledge Discovery, 265-276.
          DOI: 10.1007/3-540-45372-5_26

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

        try:
            return (
                max_distance_clusters / min_distance_clusters
            ) * sum_inverse_distances
        except ZeroDivisionError:
            return math.inf

    @staticmethod
    def _norm(x):
        origin = {i: 0 for i in x}
        return math.sqrt(utils.math.minkowski_distance(x, origin, 2))

    def update(self, x, y_pred, centers, sample_weight=1.0):

        if not self._initialized:
            self._overall_variance = {i: stats.Var() for i in x}
            self._initialized = True

        if y_pred not in self._cluster_variance:
            self._cluster_variance[y_pred] = {i: stats.Var() for i in x}

        for i in x:
            self._cluster_variance[y_pred][i].update(x[i], w=sample_weight)
            self._overall_variance[i].update(x[i], w=sample_weight)

        self._centers = centers

        return self

    def revert(self, x, y_pred, centers, sample_weight=1.0):

        for i in x:
            self._overall_variance[i].update(x[i], w=-sample_weight)
            self._cluster_variance[y_pred][i].update(x[i], w=-sample_weight)

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

        try:
            return scat_nc + dispersion_nc
        except ZeroDivisionError:
            return math.inf

    @property
    def bigger_is_better(self):
        return False
