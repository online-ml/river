import math

from river import utils

from . import base


class Silhouette(base.InternalMetric):
    """
    Silhouette coefficient [^1], roughly speaking, is the ratio between cohesion and the average distances
    from the points to their second-closest centroid. It rewards the clustering algorithm where
    points are very close to their assigned centroids and far from any other centroids,
    that is, clustering results with good cohesion and good separation.

    It rewards clusterings where points are very close to their assigned centroids and far from any other
    centroids, that is clusterings with good cohesion and good separation. [^2]

    The definition of Silhouette coefficient for online clustering evaluation is different from that of
    batch learning. It does not store information and calculate pairwise distances between all points at the
    same time, since the practice is too expensive for an incremental metric.

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
    >>> metric = metrics.cluster.Silhouette()

    >>> for x, _ in stream.iter_array(X):
    ...     k_means = k_means.learn_one(x)
    ...     y_pred = k_means.predict_one(x)
    ...     metric = metric.update(x, y_pred, k_means.centers)

    >>> metric
    Silhouette: 0.453723

    References
    ----------

    [^1]: Rousseeuw, P. (1987). Silhouettes: a graphical aid to the intepretation and validation
          of cluster analysis 20, 53 - 65. DOI: 10.1016/0377-0427(87)90125-7

    [^2]: Bifet, A. et al. (2018). "Machine Learning for Data Streams".
          DOI: 10.7551/mitpress/10654.001.0001.

    """

    def __init__(self):
        super().__init__()
        self._sum_distance_closest_centroid = 0
        self._sum_distance_second_closest_centroid = 0

    @staticmethod
    def _find_distance_second_closest_center(centers, x):
        distances = {
            i: math.sqrt(utils.math.minkowski_distance(centers[i], x, 2))
            for i in centers
        }
        return sorted(distances.values())[-2]

    def update(self, x, y_pred, centers, sample_weight=1.0):

        distance_closest_centroid = math.sqrt(
            utils.math.minkowski_distance(centers[y_pred], x, 2)
        )
        self._sum_distance_closest_centroid += distance_closest_centroid

        distance_second_closest_centroid = self._find_distance_second_closest_center(
            centers, x
        )
        self._sum_distance_second_closest_centroid += distance_second_closest_centroid

        return self

    def revert(self, x, y_pred, centers, sample_weight=1.0):

        distance_closest_centroid = math.sqrt(
            utils.math.minkowski_distance(centers[y_pred], x, 2)
        )
        self._sum_distance_closest_centroid -= distance_closest_centroid

        distance_second_closest_centroid = self._find_distance_second_closest_center(
            centers, x
        )
        self._sum_distance_second_closest_centroid -= distance_second_closest_centroid

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
