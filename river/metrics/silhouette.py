import math

from river import utils

from . import base_internal_clustering

__all__ = ["Silhouette"]


class Silhouette(base_internal_clustering.InternalClusteringMetrics):
    """
    Silhouette coefficient, roughly speaking, is the ratio between cohesion and the average distances
    from the points to their second-closest centroid. It rewards the clustering algorithm where
    points are very close to their assigned centroids and far from any other centroids,
    that is clustering results with good cohesion and good separation.

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
    >>> metric = metrics.Silhouette()

    >>> for x, _ in stream.iter_array(X):
    ...     k_means = k_means.learn_one(x)
    ...     y_pred = k_means.predict_one(x)
    ...     metric = metric.update(k_means.centers, x, y_pred)

    >>> metric
    Silhouette: 0.453723

    """

    def __init__(self):
        super().__init__()
        self._sum_distance_closest_centroid = 0
        self._sum_distance_second_closest_centroid = 0
        self.sample_correction = {}

    @staticmethod
    def _find_distance_second_closest_center(centers, point):
        distances = {
            i: math.sqrt(utils.math.minkowski_distance(centers[i], point, 2))
            for i in centers
        }
        return sorted(distances.values())[-2]

    def update(self, centers, point, y_pred, sample_weight=1.0):
        distance_closest_centroid = math.sqrt(
            utils.math.minkowski_distance(centers[y_pred], point, 2)
        )
        self._sum_distance_closest_centroid += distance_closest_centroid

        distance_second_closest_centroid = self._find_distance_second_closest_center(
            centers, point
        )
        self._sum_distance_second_closest_centroid += distance_second_closest_centroid

        # To trace back
        self.sample_correction = {
            "distance_closest_centroid": distance_closest_centroid,
            "distance_second_closest_centroid": distance_second_closest_centroid,
        }

        return self

    def revert(self, centers, point, y_pred, sample_weight=1.0, correction=None):
        self._sum_distance_closest_centroid -= correction["distance_closest_centroid"]
        self._sum_distance_second_closest_centroid -= correction[
            "distance_second_closest_centroid"
        ]

        return self

    def get(self):
        try:
            return (
                self._sum_distance_closest_centroid
                / self._sum_distance_second_closest_centroid
            )
        except ZeroDivisionError:
            return math.inf

    @property
    def bigger_is_better(self):
        return False
