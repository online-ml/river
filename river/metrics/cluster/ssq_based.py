import math

from river import metrics

from . import base

__all__ = ["CalinskiHarabasz", "Hartigan", "WB"]


class CalinskiHarabasz(base.InternalMetric):
    """Calinski-Harabasz index (CH).

    The Calinski-Harabasz index (CH) index measures the criteria simultaneously
    with the help of average between and within cluster sum of squares.

        * The **numerator** reflects the degree of separation in the way of how much centers are spread.

        * The **denominator** corresponds to compactness, to reflect how close the in-cluster objects
    are gathered around the cluster center.

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
    >>> metric = metrics.cluster.CalinskiHarabasz()

    >>> for x, _ in stream.iter_array(X):
    ...     k_means = k_means.learn_one(x)
    ...     y_pred = k_means.predict_one(x)
    ...     metric = metric.update(x, y_pred, k_means.centers)

    >>> metric
    CalinskiHarabasz: 6.922666

    References
    ----------
    [^1]: Calinski, T., Harabasz, J.-A. (1974). A Dendrite Method for Cluster Analysis.
          Communications in Statistics 3(1), 1 - 27. DOI: 10.1080/03610927408827101

    """

    def __init__(self):
        super().__init__()
        self._ssb = metrics.cluster.SSB()
        self._ssw = metrics.cluster.SSW()
        self._n_clusters = 0
        self._n_points = 0

    def update(self, x, y_pred, centers, sample_weight=1.0):

        self._ssb.update(x, y_pred, centers, sample_weight)

        self._ssw.update(x, y_pred, centers, sample_weight)

        self._n_clusters = len(centers)

        self._n_points += 1

        return self

    def revert(self, x, y_pred, centers, sample_weight=1.0):

        self._ssb.revert(x, y_pred, centers, sample_weight)

        self._ssw.revert(x, y_pred, centers, sample_weight)

        self._n_clusters = len(centers)

        self._n_points -= 1

        return self

    def get(self):
        try:
            return (self._ssb.get() / (self._n_clusters - 1)) / (
                self._ssw.get() / (self._n_points - self._n_clusters)
            )
        except ZeroDivisionError:
            return -math.inf

    @property
    def bigger_is_better(self):
        return True


class Hartigan(base.InternalMetric):
    """Hartigan Index (H - Index)

    Hartigan Index (H - Index) [^1] is a sum-of-square based index [^2], which is
    equal to the negative log of the division of SSW (Sum-of-Squares Within Clusters)
    by SSB (Sum-of-Squares Between Clusters).

    The higher the Hartigan index, the higher the clustering quality is.

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
    >>> metric = metrics.cluster.Hartigan()

    >>> for x, _ in stream.iter_array(X):
    ...     k_means = k_means.learn_one(x)
    ...     y_pred = k_means.predict_one(x)
    ...     metric = metric.update(x, y_pred, k_means.centers)

    >>> metric
    Hartigan: 0.836189

    References
    ----------
    [^1]: Hartigan JA (1975). Clustering Algorithms. John Wiley & Sons, Inc.,
          New York, NY, USA. ISBN 047135645X.

    [^2]: Q. Zhao, M. Xu, and P. Franti, "Sum-of-squares based cluster validity index
          and significance analysis," in Adaptive and Natural Computing Algorithms,
          M. Kolehmainen, P. Toivanen, and B. Beliczynski, Eds.
          Berlin, Germany: Springer, 2009, pp. 313–322.

    """

    def __init__(self):
        super().__init__()
        self._ssb = metrics.cluster.SSB()
        self._ssw = metrics.cluster.SSW()

    def update(self, x, y_pred, centers, sample_weight=1.0):

        self._ssb.update(x, y_pred, centers, sample_weight)

        self._ssw.update(x, y_pred, centers, sample_weight)

        return self

    def revert(self, x, y_pred, centers, sample_weight=1.0):

        self._ssb.revert(x, y_pred, centers, sample_weight)

        self._ssw.revert(x, y_pred, centers, sample_weight)

        return self

    def get(self):

        try:
            return -math.log(self._ssw.get() / self._ssb.get())
        except ZeroDivisionError:
            return -math.inf

    @property
    def bigger_is_better(self):
        return True


class WB(base.InternalMetric):
    """WB Index

    WB Index is a simple sum-of-square method, calculated by dividing the within
    cluster sum-of-squares by the between cluster sum-of-squares. Its effect is emphasized
    by multiplying the number of clusters. The advantages of the proposed method are
    that one can determine the number of clusters by minimizing the WB value, without
    relying on any knee point detection, and this metric is straightforward to implement.

    The lower the WB index, the higher the clustering quality is.

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
    >>> metric = metrics.cluster.WB()

    >>> for x, _ in stream.iter_array(X):
    ...     k_means = k_means.learn_one(x)
    ...     y_pred = k_means.predict_one(x)
    ...     metric = metric.update(x, y_pred, k_means.centers)

    >>> metric
    WB: 1.300077

    References
    ----------
    [^1]: Q. Zhao, M. Xu, and P. Franti, "Sum-of-squares based cluster validity index
          and significance analysis," in Adaptive and Natural Computing Algorithms,
          M. Kolehmainen, P. Toivanen, and B. Beliczynski, Eds.
          Berlin, Germany: Springer, 2009, pp. 313–322.

    """

    def __init__(self):
        super().__init__()
        self._ssb = metrics.cluster.SSB()
        self._ssw = metrics.cluster.SSW()
        self._n_clusters = 0

    def update(self, x, y_pred, centers, sample_weight=1.0):

        self._ssb.update(x, y_pred, centers, sample_weight)

        self._ssw.update(x, y_pred, centers, sample_weight)

        self._n_clusters = len(centers)

        return self

    def revert(self, x, y_pred, centers, sample_weight=1.0):

        self._ssb.revert(x, y_pred, centers, sample_weight)

        self._ssw.revert(x, y_pred, centers, sample_weight)

        self._n_clusters = len(centers)

        return self

    def get(self):

        try:
            return self._n_clusters * self._ssw.get() / self._ssb.get()
        except ZeroDivisionError:
            return math.inf

    @property
    def bigger_is_better(self):
        return False
