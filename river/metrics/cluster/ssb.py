from river import stats, utils

from . import base


class SSB(base.InternalMetric):
    """Sum-of-Squares Between Clusters (SSB).

    The Sum-of-Squares Between Clusters is the weighted mean of the squares of distances
    between cluster centers to the mean value of the whole dataset.

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
    >>> metric = metrics.cluster.SSB()

    >>> for x, _ in stream.iter_array(X):
    ...     k_means = k_means.learn_one(x)
    ...     y_pred = k_means.predict_one(x)
    ...     metric = metric.update(x, y_pred, k_means.centers)

    >>> metric
    SSB: 8.109389

    References
    ----------
    [^1]: Q. Zhao, M. Xu, and P. Franti, "Sum-of-squares based cluster validity index
          and significance analysis," in Adaptive and Natural Computing Algorithms,
          M. Kolehmainen, P. Toivanen, and B. Beliczynski, Eds.
          Berlin, Germany: Springer, 2009, pp. 313–322.

    """

    def __init__(self):
        super().__init__()
        self._center_all_points = {}
        self._n_points = 0
        self._n_points_by_clusters = {}
        self._squared_distances = {}
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

        self._n_points += 1

        try:
            self._n_points_by_clusters[y_pred] += 1
        except KeyError:
            self._n_points_by_clusters[y_pred] = 1

        for i in centers:
            self._squared_distances[i] = utils.math.minkowski_distance(
                centers[i], center_all_points, 2
            )

        return self

    def revert(self, x, y_pred, centers, sample_weight=1.0):

        for i in self._center_all_points:
            self._center_all_points[i].update(x[i], w=-sample_weight)
        center_all_points = {
            i: self._center_all_points[i].get() for i in self._center_all_points
        }

        self._n_points -= 1

        self._n_points_by_clusters[y_pred] -= 1

        for i in centers:
            self._squared_distances[i] = utils.math.minkowski_distance(
                centers[i], center_all_points, 2
            )

        return self

    def get(self):
        ssb = 0
        for i in self._n_points_by_clusters:
            try:
                ssb += (
                    1
                    / self._n_points
                    * self._n_points_by_clusters[i]
                    * self._squared_distances[i]
                )
            except ZeroDivisionError:
                ssb += 0

        return ssb

    @property
    def bigger_is_better(self):
        return True
