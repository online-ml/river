import math

from river import utils

from . import base

__all__ = ["SSQ", "Cohesion"]


class SSQ(base.MeanInternalMetric):
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
    >>> metric = metrics.cluster.SSQ()

    >>> for x, _ in stream.iter_array(X):
    ...     k_means = k_means.learn_one(x)
    ...     y_pred = k_means.predict_one(x)
    ...     metric = metric.update(x, y_pred, k_means.centers)

    >>> metric
    SSQ: 3.514277

    References
    ----------
    [^1]: Bifet, A. et al. (2018). "Machine Learning for Data Streams".
          DOI: 10.7551/mitpress/10654.001.0001.

    """

    @property
    def bigger_is_better(self):
        return False

    def _eval(self, x, y_pred, centers):
        return utils.math.minkowski_distance(centers[y_pred], x, 2)


class Cohesion(base.MeanInternalMetric):
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
    >>> metric = metrics.cluster.Cohesion()

    >>> for x, _ in stream.iter_array(X):
    ...     k_means = k_means.learn_one(x)
    ...     y_pred = k_means.predict_one(x)
    ...     metric = metric.update(x, y_pred, k_means.centers)

    >>> metric
    Cohesion: 1.682748

    References
    ----------
    [^1]: Bifet, A. et al. (2018). "Machine Learning for Data Streams".
          DOI: 10.7551/mitpress/10654.001.0001.

    """

    @property
    def bigger_is_better(self):
        return False

    def _eval(self, x, y_pred, centers):
        return math.sqrt(utils.math.minkowski_distance(centers[y_pred], x, 2))
