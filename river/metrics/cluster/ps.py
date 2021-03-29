import math

from river import stats, utils

from . import base


class PS(base.InternalMetric):
    r"""Partition Separation (PS).

    The PS index [^1] was originally developed for fuzzy clustering. This index
    only comprises a measure of separation between prototypes. Although classified
    as a batch clustering validity index (CVI), it can be readily used to evaluate
    the partitions idenified by unsupervised incremental learners tha model clusters
    using cenroids.

    Larger values of PS indicate better clustering solutions.

    The PS value is given by

    $$
    PS = \sum_{i=1}^k PS_i,
    $$

    where

    $$
    PS_i = \frac{n_i}{\max_j n_j} - exp \left[ - \frac{\min{i \neq j} (\lVert v_i - v_j \rVert_2^2)}{\beta_T} \right],
    $$

    $$
    \beta_T = \frac{1}{k} \sum_{l=1}^k \lVert v_l - \bar{v} \rVert_2 ^2,
    $$

    and

    $$
    \bar{v} = \frac{1}{k} \sum_{l=1}^k v_l.
    $$

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
    >>> metric = metrics.cluster.PS()

    >>> for x, _ in stream.iter_array(X):
    ...     k_means = k_means.learn_one(x)
    ...     y_pred = k_means.predict_one(x)
    ...     metric = metric.update(x, y_pred, k_means.centers)

    >>> metric
    PS: 1.336026

    References
    ----------
    [^1]: E. Lughofer, "Extensions of vector quantization for incremental clustering,"
          Pattern Recognit., vol. 41, no. 3, pp. 995–1011, Mar. 2008.

    """

    def __init__(self):
        super().__init__()
        self._minimum_separation = 0
        self._center_centers = {}
        self._n_points_by_cluster = {}
        self._beta_t = 0
        self._n_clusters = 0

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

        self._minimum_separation = self._find_minimum_separation(centers)

        self._center_centers = {i: stats.Mean() for i in x}

        for i in self._center_centers:
            for j in centers:
                self._center_centers[i].update(centers[j][i], w=sample_weight)

        center_centers = {
            i: self._center_centers[i].get() for i in self._center_centers
        }
        beta_t = stats.Mean()
        for i in centers:
            beta_t.update(utils.math.minkowski_distance(centers[i], center_centers, 2))
        self._beta_t = beta_t.get()

        try:
            self._n_points_by_cluster[y_pred] += 1
        except KeyError:
            self._n_points_by_cluster[y_pred] = 1

        self._n_clusters = len(centers)

        return self

    def revert(self, x, y_pred, centers, sample_weight=1.0):

        self._minimum_separation = self._find_minimum_separation(centers)

        self._center_centers = {i: stats.Mean() for i in x}

        for i in self._center_centers:
            for j in centers:
                self._center_centers[i].update(centers[j][i], w=sample_weight)

        center_centers = {
            i: self._center_centers[i].get() for i in self._center_centers
        }
        beta_t = stats.Mean()
        for i in centers:
            beta_t.update(utils.math.minkowski_distance(centers[i], center_centers, 2))
        self._beta_t = beta_t.get()

        self._n_points_by_cluster[y_pred] -= 1

        self._n_clusters = len(centers)

        return self

    def get(self):

        try:
            return sum(self._n_points_by_cluster.values()) / max(
                self._n_points_by_cluster.values()
            ) - self._n_clusters * math.exp(-self._minimum_separation / self._beta_t)
        except (ZeroDivisionError, ValueError):
            return -math.inf

    @property
    def bigger_is_better(self):
        return True
