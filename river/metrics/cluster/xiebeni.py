import math

from river import utils

from . import base


class XieBeni(base.InternalMetric):
    """Xie-Beni index (XB).

    The Xie-Beni index [^1] has the form of (Compactness)/(Separation), which defines the
    inter-cluster separation as the minimum squared distance between cluster centers,
    and the intra-cluster compactness as the mean squared distance between each data
    object and its cluster centers. The smaller the value of XB, the better the
    clustering quality.

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
    >>> metric = metrics.cluster.XieBeni()

    >>> for x, _ in stream.iter_array(X):
    ...     k_means = k_means.learn_one(x)
    ...     y_pred = k_means.predict_one(x)
    ...     metric = metric.update(x, y_pred, k_means.centers)

    >>> metric
    XieBeni: 0.397043

    References
    ----------

    [^1]: X. L. Xie, G. Beni (1991). A validity measure for fuzzy clustering. In: IEEE
          Transactions on Pattern Analysis and Machine Intelligence 13(8), 841 - 847.
          DOI: 10.1109/34.85677

    """

    def __init__(self):
        super().__init__()
        self._ssw = 0
        self._minimum_separation = 0
        self._total_points = 0

    @staticmethod
    def _find_minimum_separation(centers):
        minimum_separation = math.inf
        n_centers = max(centers) + 1
        for i in range(n_centers):
            for j in range(i + 1, n_centers):
                separation_ij = utils.math.minkowski_distance(centers[i], centers[j], 2)
                if separation_ij < minimum_separation:
                    minimum_separation = separation_ij
        return minimum_separation

    def update(self, x, y_pred, centers, sample_weight=1.0):

        squared_distance = utils.math.minkowski_distance(centers[y_pred], x, 2)
        minimum_separation = self._find_minimum_separation(centers)

        self._ssw += squared_distance
        self._total_points += 1
        self._minimum_separation = minimum_separation

        return self

    def revert(self, x, y_pred, centers, sample_weight=1.0):

        squared_distance = utils.math.minkowski_distance(centers[y_pred], x, 2)
        minimum_separation = self._find_minimum_separation(centers)

        self._ssw -= squared_distance
        self._total_points -= 1
        self._minimum_separation -= minimum_separation

        return self

    def get(self):
        try:
            return self._ssw / (self._total_points * self._minimum_separation)
        except ZeroDivisionError:
            return math.inf

    @property
    def bigger_is_better(self):
        return False
