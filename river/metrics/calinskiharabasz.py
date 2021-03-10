from river import stats, utils

from . import base_internal_clustering

__all__ = ["CalinskiHarabasz"]


class CalinskiHarabasz(base_internal_clustering.InternalClusteringMetrics):
    """Calinski-Harabasz index (CH).

    The Calinski-Harabasz index (CH) index measures the two criteria simultaneously
    with the help of average between and within cluster sum of squares. The numerator
    reflects the degree of separation in the way of how much centers are spread, and
    the denominator corresponds to compactness, to reflect how close the in-cluster objects
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
    >>> metric = metrics.CalinskiHarabasz()

    >>> for x, _ in stream.iter_array(X):
    ...     k_means = k_means.learn_one(x)
    ...     y_pred = k_means.predict_one(x)
    ...     metric = metric.update(x, y_pred, k_means.centers)

    >>> metric
    CalinskiHarabasz: 2.540276

    References
    ----------
    [^1]: Calinski, T., Harabasz, J.-A. (1974). A Dendrite Method for Cluster Analysis.
          Communications in Statistics 3(1), 1 - 27. DOI: 10.1080/03610927408827101

    """

    def __init__(self):
        super().__init__()
        self._center_all_points = {}
        self._ssq_points_centers = 0
        self._ssq_centers_center = 0
        self._n_points = 0
        self._n_clusters = 0
        self.sample_correction = {}
        self._initialized = False

    def update(self, x, y_pred, centers, sample_weight=1.0):

        squared_distance_point_center = utils.math.minkowski_distance(
            centers[y_pred], x, 2
        )

        if not self._initialized:
            self._center_all_points = {i: stats.Mean() for i in x}
            self._initialized = True

        # To trace back
        self.sample_correction = {
            "squared_distance_point_center": squared_distance_point_center
        }

        for i in self._center_all_points:
            self._center_all_points[i].update(x[i], w=sample_weight)
        center_all_points = {
            i: self._center_all_points[i].get() for i in self._center_all_points
        }
        self._ssq_points_centers += squared_distance_point_center
        ssq_centers_center = 0
        for i in centers:
            ssq_centers_center += utils.math.minkowski_distance(
                centers[i], center_all_points, 2
            )
        self._ssq_centers_center = ssq_centers_center
        self._n_points += 1
        self._n_clusters = len(centers)

        return self

    def revert(self, x, y_pred, centers, sample_weight=1.0, correction=None):

        for i in self._center_all_points:
            self._center_all_points[i].update(x[i], w=-sample_weight)
        center_all_points = {
            i: self._center_all_points[i].get() for i in self._center_all_points
        }
        ssq_centers_center = 0
        for i in centers:
            ssq_centers_center += utils.math.minkowski_distance(
                centers[i], center_all_points, 2
            )
        self._ssq_centers_center = ssq_centers_center
        self._ssq_points_centers -= correction["squared_distance_point_center"]
        self._n_points -= 1
        self._n_clusters = len(centers)

        return self

    def get(self):
        return (self._ssq_centers_center / (self._n_clusters - 1)) / (
            self._ssq_points_centers / (self._n_points - self._n_clusters)
        )

    @property
    def bigger_is_better(self):
        return False
