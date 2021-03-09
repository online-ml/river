import math

from river import utils

from . import base_internal_clustering

__all__ = ["Cohesion", "MSSTD", "RMSSTD", "SSQ", "Separation", "Silhouette", "XieBeni"]


class Cohesion(base_internal_clustering.MeanInternalMetric):
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
    ...     [4, 0]
    ... ]

    >>> k_means = cluster.KMeans(n_clusters=2, halflife=0.4, sigma=3, seed=0)
    >>> metric = metrics.Cohesion()

    >>> for x, _ in stream.iter_array(X):
    ...     k_means = k_means.learn_one(x)
    ...     y_pred = k_means.predict_one(x)
    ...     metric = metric.update(k_means.centers, x, y_pred)

    >>> metric
    Cohesion: 1.975643

    """

    @property
    def bigger_is_better(self):
        return False

    def _eval(self, centers, point, y_pred):
        return math.sqrt(utils.math.minkowski_distance(centers[y_pred], point, 2))


class SSQ(base_internal_clustering.MeanInternalMetric):
    """Mean of sum of squared (SSQ) distances from data points to their assigned cluster centroids.
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
    ...     [4, 0]
    ... ]

    >>> k_means = cluster.KMeans(n_clusters=2, halflife=0.4, sigma=3, seed=0)
    >>> metric = metrics.SSQ()

    >>> for x, _ in stream.iter_array(X):
    ...     k_means = k_means.learn_one(x)
    ...     y_pred = k_means.predict_one(x)
    ...     metric = metric.update(k_means.centers, x, y_pred)

    >>> metric
    SSQ: 4.226734

    """

    @property
    def bigger_is_better(self):
        return False

    def _eval(self, centers, point, y_pred):
        return utils.math.minkowski_distance(centers[y_pred], point, 2)


class MSSTD(base_internal_clustering.InternalClusteringMetrics):
    """Mean squared standard deviation.

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
    >>> metric = metrics.MSSTD()

    >>> for x, _ in stream.iter_array(X):
    ...     k_means = k_means.learn_one(x)
    ...     y_pred = k_means.predict_one(x)
    ...     metric = metric.update(k_means.centers, x, y_pred)

    >>> metric
    MSSTD: 2.635708

    """

    def __init__(self):
        super().__init__()
        self._ssq = 0
        self._total_points = 0
        self._total_clusters = 0
        self._dim = 0
        self.sample_correction = {}

    def update(self, centers, point, y_pred, sample_weight=1.0):

        squared_distance = utils.math.minkowski_distance(centers[y_pred], point, 2)
        n_added_centers = len(centers) - self._total_clusters

        # To trace back
        self.sample_correction = {
            "squared_distance": squared_distance,
            "n_added_centers": n_added_centers,
        }

        self._ssq += squared_distance
        self._total_points += 1
        self._total_clusters += n_added_centers
        self._dim = len(point)

        return self

    def revert(self, centers, point, y_pred, sample_weight=1.0, correction=None):
        self._ssq -= correction["squared_distance"]
        self._total_clusters -= correction["n_added_centers"]
        self._total_points -= 1

        return self

    def get(self):
        try:
            return self._ssq / (self._dim * (self._total_points - self._total_clusters))
        except ZeroDivisionError:
            return math.inf

    @property
    def bigger_is_better(self):
        return False


class RMSSTD(MSSTD):
    """Root mean squared standard deviation.

    This is the square root of the pooled sample variance of all the attributes, which
    measures only the compactness of found clusters.

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
    >>> metric = metrics.MSSTD()

    >>> for x, _ in stream.iter_array(X):
    ...     k_means = k_means.learn_one(x)
    ...     y_pred = k_means.predict_one(x)
    ...     metric = metric.update(k_means.centers, x, y_pred)

    >>> metric
    RMSSTD: 1.623486

    """

    def get(self):
        return super().get() ** 0.5


class Separation(base_internal_clustering.MeanInternalMetric):
    """Average distance from a point to the points assigned to other clusters. The bigger the better.

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
    ...     [4, 0]
    ... ]

    >>> k_means = cluster.KMeans(n_clusters=2, halflife=0.4, sigma=3, seed=0)
    >>> metric = metrics.Separation()

    >>> for x, _ in stream.iter_array(X):
    ...     k_means = k_means.learn_one(x)
    ...     y_pred = k_means.predict_one(x)
    ...     metric = metric.update(k_means.centers, x, y_pred)

    >>> metric
    Separation: 4.647488

    """

    @property
    def bigger_is_better(self):
        return True

    def _eval(self, centers, point, y_pred):
        sum_distance_other_clusters = 0
        for i in centers:
            if i != y_pred:
                sum_distance_other_clusters += math.sqrt(
                    utils.math.minkowski_distance(centers[i], point, 2)
                )
        return sum_distance_other_clusters / (len(centers) - 1)


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


class XieBeni(base_internal_clustering.InternalClusteringMetrics):
    """Xie-Beni index (XB).

    The Xie-Beni index has the form of (Compactness)/(Separation), which defines the
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
    >>> metric = metrics.MSSTD()

    >>> for x, _ in stream.iter_array(X):
    ...     k_means = k_means.learn_one(x)
    ...     y_pred = k_means.predict_one(x)
    ...     metric = metric.update(k_means.centers, x, y_pred)

    >>> metric
    MSSTD: 2.635708

    """

    def __init__(self):
        super().__init__()
        self._ssq = 0
        self._minimum_separation = 0
        self._total_points = 0
        self.sample_correction = {}

    @staticmethod
    def _find_minimum_separation(centers):
        minimum_separation = math.inf
        n_centers = max(centers)
        for i in range(n_centers):
            for j in range(i, n_centers):
                separation_ij = math.sqrt(
                    utils.math.minkowski_distance(centers[i], centers[j], 2)
                )
                if separation_ij < minimum_separation:
                    minimum_separation = separation_ij
        return minimum_separation

    def update(self, centers, point, y_pred, sample_weight=1.0):

        squared_distance = utils.math.minkowski_distance(centers[y_pred], point, 2)
        minimum_separation = self._find_minimum_separation(centers)

        # To trace back
        self.sample_correction = {
            "squared_distance": squared_distance,
            "separation_difference": minimum_separation - self._minimum_separation,
        }

        self._ssq += squared_distance
        self._total_points += 1
        self._minimum_separation += minimum_separation - self._minimum_separation

        return self

    def revert(self, centers, point, y_pred, sample_weight=1.0, correction=None):
        self._ssq -= correction["squared_distance"]
        self._total_points -= 1
        self._minimum_separation -= correction["separation_difference"]

        return self

    def get(self):
        try:
            return self._ssq / (self._total_points * self._minimum_separation)
        except ZeroDivisionError:
            return math.inf

    @property
    def bigger_is_better(self):
        return False
