import math

from river import metrics, utils

from . import base

__all__ = ["BallHall", "Cohesion", "SSW", "Xu"]


class SSW(base.MeanInternalMetric):
    """Sum-of-Squares Within Clusters (SSW).

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
    >>> metric = metrics.cluster.SSW()

    >>> for x, _ in stream.iter_array(X):
    ...     k_means = k_means.learn_one(x)
    ...     y_pred = k_means.predict_one(x)
    ...     metric = metric.update(x, y_pred, k_means.centers)

    >>> metric
    SSW: 3.514277

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


class BallHall(base.InternalMetric):
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
        self._ssw = metrics.cluster.SSW()
        self._n_clusters = 0

    def update(self, x, y_pred, centers, sample_weight=1.0):

        self._ssw.update(x, y_pred, centers, sample_weight)

        self._n_clusters = len(centers)

        return self

    def revert(self, x, y_pred, centers, sample_weight=1.0):

        self._ssw.revert(x, y_pred, centers, sample_weight)

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


class Xu(base.InternalMetric):
    """Xu Index

    Xu Index is among the most complicated sum-of-squares based metrics [^1].
    It is calculated based on the Sum-of-Squares Within Clusters (SSW), total
    number of points, number of clusters, and the dimension of the cluserting problem.

    The lower the Xu index, the higher the clustering quality is.

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
    >>> metric = metrics.cluster.Xu()

    >>> for x, _ in stream.iter_array(X):
    ...     k_means = k_means.learn_one(x)
    ...     y_pred = k_means.predict_one(x)
    ...     metric = metric.update(x, y_pred, k_means.centers)

    >>> metric
    Xu: -2.73215

    References
    ----------
    [^1]: Q. Zhao, M. Xu, and P. Franti, "Sum-of-squares based cluster validity index
          and significance analysis," in Adaptive and Natural Computing Algorithms,
          M. Kolehmainen, P. Toivanen, and B. Beliczynski, Eds.
          Berlin, Germany: Springer, 2009, pp. 313–322.

    """

    def __init__(self):
        super().__init__()
        self._ssw = metrics.cluster.SSW()
        self._n_points = 0
        self._n_clusters = 0
        self._dim = 0
        self._initialized = False

    def update(self, x, y_pred, centers, sample_weight=1.0):

        if not self._initialized:
            self._dim = len(x)

        if len(x) == self._dim:

            self._ssw.update(x, y_pred, centers, sample_weight)

            self._n_points += 1

            self._n_clusters = len(centers)

        return self

    def revert(self, x, y_pred, centers, sample_weight=1.0):

        if len(x) == self._dim:

            self._ssw.revert(x, y_pred, centers, sample_weight)

            self._n_points -= 1

            self._n_clusters = len(centers)

        return self

    def get(self):

        try:
            return self._dim * math.log(
                math.sqrt(
                    self._ssw.get() / (self._dim * (self._n_points * self._n_points))
                )
            ) + math.log(self._n_clusters)
        except ZeroDivisionError:
            return math.inf

    @property
    def bigger_is_better(self):
        return False
