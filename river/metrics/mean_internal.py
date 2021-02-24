import math

from river import utils

from . import base_internal_clustering

__all__ = ["Cohesion", "SSQ"]


class Cohesion(base_internal_clustering.MeanMetric):
    """Mean distance from the points to their assigned cluster centroids.

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
    ...     [4, 0]
    ... ]

    >>> k_means = cluster.KMeans(n_clusters=2, halflife=0.4, sigma=3, seed=0)
    >>> metric = metrics.Cohesion()

    >>> for x, _ in enumerate(stream.iter_array(X)):
    ...     k_means = k_means.learn_one(x)
    ...     label = k_means.predict_one(x)
    ...     metric = metric.update(k_means, x, label)

    >>> metric

    """

    def _eval(self, method, point, label):
        return math.sqrt(utils.math.minkowski_distance(method.centers[label], point, 2))


class SSQ(Cohesion):
    """ Mean of sum of squared (SSQ) distances from data points to their assigned cluster centroids.

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
    ...     [4, 0]
    ... ]

    >>> k_means = cluster.KMeans(n_clusters=2, halflife=0.4, sigma=3, seed=0)
    >>> metric = metrics.SSQ()

    >>> for x, _ in stream.iter_array(X):
    ...     k_means = k_means.learn_one(x)
    ...     label = k_means.predict_one(x)
    ...     metric = metric.update(k_means, x, label)

    >>> metric

    """

    def get(self):
        return super().get() * super().get()
