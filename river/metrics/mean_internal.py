import math

from river import utils

from . import base_internal_clustering

__all__ = ["Cohesion", "SSQ", "Separation"]


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
    ...     [4, 0]
    ... ]

    >>> k_means = cluster.KMeans(n_clusters=2, halflife=0.4, sigma=3, seed=0)
    >>> metric = metrics.Cohesion()

    >>> for x, _ in stream.iter_array(X):
    ...     k_means = k_means.learn_one(x)
    ...     label = k_means.predict_one(x)
    ...     metric = metric.update(k_means, x, label)

    >>> metric
    Cohesion: 1.975643
    """

    @property
    def bigger_is_better(self):
        return False

    def _eval(self, method, point, label):
        return math.sqrt(utils.math.minkowski_distance(method.centers[label], point, 2))


class SSQ(base_internal_clustering.MeanInternalMetric):
    """Mean of sum of squared (SSQ) distances from data points to their assigned cluster centroids. The bigger the better.

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
    SSQ: 3.903166
    """

    @property
    def bigger_is_better(self):
        return False

    def _eval(self, method, point, label):
        return utils.math.minkowski_distance(method.centers[label], point, 2)


class Separation(base_internal_clustering.MeanInternalMetric):
    """Average distance from a point to the points assigned to other clusters. The bigger the better.

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
    >>> metric = metrics.Separation()

    >>> for x, _ in stream.iter_array(X):
    ...     k_means = k_means.learn_one(x)
    ...     label = k_means.predict_one(x)
    ...     metric = metric.update(k_means, x, label)

    >>> metric
    Separation: 4.647488
    """

    @property
    def bigger_is_better(self):
        return True

    def _eval(self, method, point, label):
        sum_distance_other_clusters = 0
        for i in range(len(method.centers)):
            if i != label:
                sum_distance_other_clusters += math.sqrt(
                    utils.math.minkowski_distance(method.centers[i], point, 2)
                )
        return sum_distance_other_clusters / (len(method.centers) - 1)
