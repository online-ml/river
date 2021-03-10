import math

from river import utils

from . import base_internal_clustering

__all__ = ["SSQ", "Cohesion"]


class SSQ(base_internal_clustering.MeanInternalMetric):
    """Mean of sum of squared (SSQ) distances from data points to their assigned cluster centroids.
    The bigger the better.

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
    >>> metric = metrics.SSQ()

    >>> for x, _ in stream.iter_array(X):
    ...     k_means = k_means.learn_one(x)
    ...     y_pred = k_means.predict_one(x)
    ...     metric = metric.update(k_means.centers, x, y_pred)

    >>> metric
    SSQ: 3.514277

    """

    @property
    def bigger_is_better(self):
        return False

    def _eval(self, centers, point, y_pred):
        return utils.math.minkowski_distance(centers[y_pred], point, 2)


class Cohesion(base_internal_clustering.MeanInternalMetric):
    """Mean distance from the points to their assigned cluster centroids. The smaller the better.

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
    >>> metric = metrics.Cohesion()

    >>> for x, _ in stream.iter_array(X):
    ...     k_means = k_means.learn_one(x)
    ...     y_pred = k_means.predict_one(x)
    ...     metric = metric.update(k_means.centers, x, y_pred)

    >>> metric
    Cohesion: 1.682748

    """

    @property
    def bigger_is_better(self):
        return False

    def _eval(self, centers, point, y_pred):
        return math.sqrt(utils.math.minkowski_distance(centers[y_pred], point, 2))
