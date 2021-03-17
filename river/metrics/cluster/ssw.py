import math

from river import stats, utils

from . import base

__all__ = ["BallHall", "SSW", "Cohesion"]


class SSW(base.MeanInternalMetric):
    """ Sum-of-Squares Within Clusters (SSW).

    Mean of sum of squared distances from data points to their assigned cluster centroids.
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


class BallHall(base.InternalClusMetric):
    """Ball-Hall index

    Ball-Hall index is a sum-of-squared based index. It is calculated by
    dividing the sum-of-squares between clusters by the number of generated
    clusters.

    The index is usually used to evaluate the number of clusters by the following
    criteria: the maximum value of the successive difference is determined as the
    optimal number of clusters.

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
    >>> metric = metrics.cluster.BallHall()

    >>> for x, _ in stream.iter_array(X):
    ...     k_means = k_means.learn_one(x)
    ...     y_pred = k_means.predict_one(x)
    ...     metric = metric.update(x, y_pred, k_means.centers)

    >>> metric
    BallHall: 1.171426

    References
    ----------
    [^1]: Ball, G.H., Hubert, L.J.: ISODATA, A novel method of data analysis
          and pattern classification (Tech. Rep. NTIS No. AD 699616).
          Standford Research Institute, Menlo Park (1965)

    """

    def __init__(self):
        self._ssw = stats.Mean()
        self._n_clusters = 0

    def update(self, x, y_pred, centers, sample_weight=1.0):

        squared_distance = utils.math.minkowski_distance(centers[y_pred], x, 2)

        self._ssw.update(squared_distance, w=sample_weight)
        self._n_clusters = len(centers)

        return self

    def revert(self, x, y_pred, centers, sample_weight=1.0):

        squared_distance = utils.math.minkowski_distance(centers[y_pred], x, 2)

        self._ssw.update(squared_distance, w=-sample_weight)
        self._n_clusters = len(centers)

        return self

    def get(self):
        try:
            return self._ssw.get() / self._n_clusters
        except ZeroDivisionError:
            return math.inf

    @property
    def bigger_is_better(self):
        return False
