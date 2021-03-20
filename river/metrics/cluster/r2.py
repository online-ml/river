import math

from river import stats, utils

from . import base


class R2(base.InternalMetric):
    """R-Squared

    R-Squared (RS) [^1] is the complement of the ratio of sum of squared distances between objects
    in different clusters to the total sum of squares. It is an intuitive and simple formulation
    of measuring the differences between clusters.

    The maximum value of R-Squared is 1, which means that the higher the index, the better
    the clustering results.

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
    >>> metric = metrics.cluster.R2()

    >>> for x, _ in stream.iter_array(X):
    ...     k_means = k_means.learn_one(x)
    ...     y_pred = k_means.predict_one(x)
    ...     metric = metric.update(x, y_pred, k_means.centers)

    >>> metric
    R2: 0.509203

    References
    ----------
    [^1]: Halkidi, M., Vazirgiannis, M., & Batistakis, Y. (2000). Quality Scheme Assessment in the
          Clustering Process. Principles Of Data Mining And Knowledge Discovery, 265-276.
          DOI: 10.1007/3-540-45372-5_26

    """

    def __init__(self):
        super().__init__()
        self._center_all_points = {}
        self._ssq_point_center = 0
        self._ssq_point_cluster_centers = 0
        self._cluster_variance = {}
        self._centers = {}
        self._initialized = False

    def update(self, x, y_pred, centers, sample_weight=1.0):

        if not self._initialized:
            self._center_all_points = {i: stats.Mean() for i in x}
            self._initialized = True
        for i in self._center_all_points:
            self._center_all_points[i].update(x[i], w=sample_weight)
        center_all_points = {
            i: self._center_all_points[i].get() for i in self._center_all_points
        }

        squared_distance_center = utils.math.minkowski_distance(x, center_all_points, 2)
        squared_distance_cluster_center = utils.math.minkowski_distance(
            x, centers[y_pred], 2
        )

        self._ssq_point_center += squared_distance_center
        self._ssq_point_cluster_centers += squared_distance_cluster_center

        return self

    def revert(self, x, y_pred, centers, sample_weight=1.0):

        for i in self._center_all_points:
            self._center_all_points[i].update(x[i], w=-sample_weight)
        center_all_points = {
            i: self._center_all_points[i].get() for i in self._center_all_points
        }

        squared_distance_center = utils.math.minkowski_distance(x, center_all_points, 2)
        squared_distance_cluster_center = utils.math.minkowski_distance(
            x, centers[y_pred], 2
        )

        self._ssq_point_center -= squared_distance_center
        self._ssq_point_cluster_centers -= squared_distance_cluster_center

        return self

    def get(self):
        try:
            return 1 - self._ssq_point_cluster_centers / self._ssq_point_center
        except ZeroDivisionError:
            return -math.inf

    @property
    def bigger_is_better(self):
        return True
