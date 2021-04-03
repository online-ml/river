import math

from river import stats, utils

from . import base

__all__ = ["GD43", "GD53"]


class GD43(base.InternalMetric):
    r"""Generalized Dunn's index 43 (GD43).

    The Generalized Dunn's indices comprise a set of 17 variants of the original
    Dunn's index devised to address sensitivity to noise in the latter. The formula
    of this index is given by:

    $$
    GD_{rs} = \frac{\min_{i \new q} [\delta_r (\omega_i, \omega_j)]}{\max_k [\Delta_s (\omega_k)]},
    $$

    where $\delta_r(.)$ is a measure of separation, and $\Delta_s(.)$ is a measure of compactness,
    the parameters $r$ and $s$ index the measures' formulations. In particular, when employing
    Euclidean distance, GD43 is formulated using:

    $$
    \delta_4 (\omega_i, \omega_j) = \lVert v_i - v_j \rVert_2,
    $$

    and

    $$
    \Delta_3 (\omega_k) = \frac{2 \times CP_1^2 (v_k, \omega_k)}{n_k}.
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
    >>> metric = metrics.cluster.GD43()

    >>> for x, _ in stream.iter_array(X):
    ...     k_means = k_means.learn_one(x)
    ...     y_pred = k_means.predict_one(x)
    ...     metric = metric.update(x, y_pred, k_means.centers)

    >>> metric
    GD43: 0.731369

    References
    ----------
    [^1]:  J. Bezdek and N. Pal, "Some new indexes of cluster validity,"
           IEEE Trans. Syst., Man, Cybern. B, vol. 28, no. 3, pp. 301–315, Jun. 1998.

    """

    def __init__(self):
        super().__init__()
        self._minimum_separation = 0
        self._avg_cp_by_clusters = {}

    @staticmethod
    def _find_minimum_separation(centers):
        minimum_separation = math.inf
        n_centers = max(centers) + 1
        for i in range(n_centers):
            for j in range(i + 1, n_centers):
                separation_ij = math.sqrt(
                    utils.math.minkowski_distance(centers[i], centers[j], 2)
                )
                if separation_ij < minimum_separation:
                    minimum_separation = separation_ij
        return minimum_separation

    def update(self, x, y_pred, centers, sample_weight=1.0):

        self._minimum_separation = self._find_minimum_separation(centers)

        distance = math.sqrt(utils.math.minkowski_distance(centers[y_pred], x, 2))

        if y_pred in self._avg_cp_by_clusters:
            self._avg_cp_by_clusters[y_pred].update(distance, w=sample_weight)
        else:
            self._avg_cp_by_clusters[y_pred] = stats.Mean()
            self._avg_cp_by_clusters[y_pred].update(distance, w=sample_weight)

        return self

    def revert(self, x, y_pred, centers, sample_weight=1.0):

        self._minimum_separation = self._find_minimum_separation(centers)

        distance = math.sqrt(utils.math.minkowski_distance(centers[y_pred], x, 2))

        self._avg_cp_by_clusters[y_pred].update(distance, w=-sample_weight)

        return self

    def get(self):
        avg_cp_by_clusters = {
            i: self._avg_cp_by_clusters[i].get() for i in self._avg_cp_by_clusters
        }

        try:
            return self._minimum_separation / (2 * max(avg_cp_by_clusters.values()))
        except ZeroDivisionError:
            return -math.inf

    @property
    def bigger_is_better(self):
        return True


class GD53(base.InternalMetric):
    r"""Generalized Dunn's index 53 (GD53).

    The Generalized Dunn's indices comprise a set of 17 variants of the original
    Dunn's index devised to address sensitivity to noise in the latter. The formula
    of this index is given by:

    $$
    GD_{rs} = \frac{\min_{i \new q} [\delta_r (\omega_i, \omega_j)]}{\max_k [\Delta_s (\omega_k)]},
    $$

    where $\delta_r(.)$ is a measure of separation, and $\Delta_s(.)$ is a measure of compactness,
    the parameters $r$ and $s$ index the measures' formulations. In particular, when employing
    Euclidean distance, GD43 is formulated using:

    $$
    \delta_5 (\omega_i, \omega_j) = \frac{CP_1^2 (v_i, \omega_i) + CP_1^2 (v_j, \omega_j)}{n_i + n_j},
    $$

    and

    $$
    \Delta_3 (\omega_k) = \frac{2 \times CP_1^2 (v_k, \omega_k)}{n_k}.
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
    >>> metric = metrics.cluster.GD53()

    >>> for x, _ in stream.iter_array(X):
    ...     k_means = k_means.learn_one(x)
    ...     y_pred = k_means.predict_one(x)
    ...     metric = metric.update(x, y_pred, k_means.centers)

    >>> metric
    GD53: 0.158377

    References
    ----------
    [^1]:  J. Bezdek and N. Pal, "Some new indexes of cluster validity,"
           IEEE Trans. Syst., Man, Cybern. B, vol. 28, no. 3, pp. 301–315, Jun. 1998.

    """

    def __init__(self):
        super().__init__()
        self._minimum_separation = 0
        self._cp_by_clusters = {}
        self._n_points_by_clusters = {}
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

        distance = math.sqrt(utils.math.minkowski_distance(centers[y_pred], x, 2))

        try:
            self._cp_by_clusters[y_pred] += distance
            self._n_points_by_clusters[y_pred] += 1
        except KeyError:
            self._cp_by_clusters[y_pred] = distance
            self._n_points_by_clusters[y_pred] = 1

        self._n_clusters = len(centers)

        return self

    def revert(self, x, y_pred, centers, sample_weight=1.0):

        self._minimum_separation = self._find_minimum_separation(centers)

        distance = math.sqrt(utils.math.minkowski_distance(centers[y_pred], x, 2))

        self._cp_by_clusters[y_pred] -= distance
        self._n_points_by_clusters[y_pred] -= 1

        self._n_clusters = len(centers)

        return self

    def get(self):

        min_delta_5 = math.inf
        for i in range(self._n_clusters):
            for j in range(i + 1, self._n_clusters):
                try:
                    delta_5 = (self._cp_by_clusters[i] + self._cp_by_clusters[j]) / (
                        self._n_points_by_clusters[i] + self._n_points_by_clusters[j]
                    )
                except KeyError:
                    continue

                if delta_5 < min_delta_5:
                    min_delta_5 = delta_5

        try:
            return min_delta_5 / self._minimum_separation
        except ZeroDivisionError:
            return -math.inf

    @property
    def bigger_is_better(self):
        return True
