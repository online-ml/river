import math

from river import utils

from . import base


class Separation(base.MeanInternalMetric):
    """Average distance from a point to the points assigned to other clusters.
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
    >>> metric = metrics.cluster.Separation()

    >>> for x, _ in stream.iter_array(X):
    ...     k_means = k_means.learn_one(x)
    ...     y_pred = k_means.predict_one(x)
    ...     metric = metric.update(x, y_pred, k_means.centers)

    >>> metric
    Separation: 4.54563

    References
    ----------
    [^1]: Bifet, A. et al. (2018). "Machine Learning for Data Streams".
          DOI: 10.7551/mitpress/10654.001.0001.

    """

    @property
    def bigger_is_better(self):
        return True

    def _eval(self, x, y_pred, centers):
        sum_distance_other_clusters = 0
        for i in centers:
            if i != y_pred:
                sum_distance_other_clusters += math.sqrt(
                    utils.math.minkowski_distance(centers[i], x, 2)
                )
        try:
            return sum_distance_other_clusters / (len(centers) - 1)
        except ZeroDivisionError:
            return -math.inf
