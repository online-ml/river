import math

from river import utils

from . import base

__all__ = ["MSSTD", "RMSSTD"]


class MSSTD(base.InternalMetric):
    """Mean Squared Standard Deviation.

    This is the pooled sample variance of all the attributes, which measures
    only the compactness of found clusters.

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
    >>> metric = metrics.cluster.MSSTD()

    >>> for x, _ in stream.iter_array(X):
    ...     k_means = k_means.learn_one(x)
    ...     y_pred = k_means.predict_one(x)
    ...     metric = metric.update(x, y_pred, k_means.centers)

    >>> metric
    MSSTD: 2.635708

    References
    ----------
    [^1]: Halkidi, M., Batistakis, Y. and Vazirgiannis, M. (2001). On Clustering Validation Techniques.
          Journal of Intelligent Information Systems, 17, 107 - 145.
          DOI: 10.1023/a:1012801612483.

    """

    def __init__(self):
        super().__init__()
        self._ssq = 0
        self._total_points = 0
        self._total_clusters = 0
        self._dim = 0

    def update(self, x, y_pred, centers, sample_weight=1.0):

        squared_distance = utils.math.minkowski_distance(centers[y_pred], x, 2)
        n_added_centers = len(centers) - self._total_clusters

        self._ssq += squared_distance
        self._total_points += 1
        self._total_clusters += n_added_centers
        self._dim = len(x)

        return self

    def revert(self, x, y_pred, centers, sample_weight=1.0):

        squared_distance = utils.math.minkowski_distance(centers[y_pred], x, 2)
        n_added_centers = len(centers) - self._total_clusters

        self._ssq -= squared_distance
        self._total_clusters -= n_added_centers
        self._total_points -= 1

        return self

    def get(self):
        try:
            return self._ssq / (self._dim * (self._total_points - self._total_clusters))
        except ZeroDivisionError:
            return math.inf

    @property
    def bigger_is_better(self):
        return False


class RMSSTD(MSSTD):
    """Root Mean Squared Standard Deviation.

    This is the square root of the pooled sample variance of all the attributes, which
    measures only the compactness of found clusters.

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
    >>> metric = metrics.cluster.RMSSTD()

    >>> for x, _ in stream.iter_array(X):
    ...     k_means = k_means.learn_one(x)
    ...     y_pred = k_means.predict_one(x)
    ...     metric = metric.update(x, y_pred, k_means.centers)

    >>> metric
    RMSSTD: 1.623486

    References
    ----------
    [^1]: Halkidi, M., Batistakis, Y. and Vazirgiannis, M. (2001). On Clustering Validation Techniques.
          Journal of Intelligent Information Systems, 17, 107 - 145.
          DOI: 10.1023/a:1012801612483.

    """

    def get(self):
        return super().get() ** 0.5
