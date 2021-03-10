import math

from river import stats, utils

from . import base_internal_clustering

__all__ = [
    "MSSTD",
    "RMSSTD",
    "SSQ",
    "Cohesion",
    "CalinskiHarabasz",
    "DaviesBouldin",
    "IIndex",
    "Separation",
    "Silhouette",
    "XieBeni",
]


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
    >>> metric = metrics.RMSSTD()

    >>> for x, _ in stream.iter_array(X):
    ...     k_means = k_means.learn_one(x)
    ...     y_pred = k_means.predict_one(x)
    ...     metric = metric.update(k_means.centers, x, y_pred)

    >>> metric
    RMSSTD: 1.623486

    """

    def get(self):
        return super().get() ** 0.5


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
    ...     [4, 0],
    ...     [-2, 2],
    ...     [-2, 4],
    ...     [-2, 0]
    ... ]

    >>> k_means = cluster.KMeans(n_clusters=3, halflife=0.4, sigma=3, seed=0)
    >>> metric = metrics.SSQ()

    >>> for x, _ in stream.iter_array(X):
    ...     k_means = k_means.learn_one(x)
    ...     y_pred = k_means.predict_one(x)
    ...     metric = metric.update(k_means.centers, x, y_pred)

    >>> metric
    SSQ: 3.514277

    """

    @property
    def bigger_is_better(self):
        return False

    def _eval(self, centers, point, y_pred):
        return utils.math.minkowski_distance(centers[y_pred], point, 2)


class CalinskiHarabasz(base_internal_clustering.InternalClusteringMetrics):
    """Calinski-Harabasz index (CH).

    The Davies-Bouldin index (DB) index measures the two criteria simultaneously
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
    ...     metric = metric.update(k_means.centers, x, y_pred)

    >>> metric
    CalinskiHarabasz: 2.540276

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

    def update(self, centers, point, y_pred, sample_weight=1.0):

        squared_distance_point_center = utils.math.minkowski_distance(
            centers[y_pred], point, 2
        )

        if not self._initialized:
            self._center_all_points = {i: stats.Mean() for i in point}
            self._initialized = True

        # To trace back
        self.sample_correction = {
            "squared_distance_point_center": squared_distance_point_center
        }

        for i in self._center_all_points:
            self._center_all_points[i].update(point[i], w=sample_weight)
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

    def revert(self, centers, point, y_pred, sample_weight=1.0, correction=None):

        for i in self._center_all_points:
            self._center_all_points[i].update(point[i], w=-sample_weight)
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
    ...     [4, 0],
    ...     [-2, 2],
    ...     [-2, 4],
    ...     [-2, 0]
    ... ]

    >>> k_means = cluster.KMeans(n_clusters=3, halflife=0.4, sigma=3, seed=0)
    >>> metric = metrics.Cohesion()

    >>> for x, _ in stream.iter_array(X):
    ...     k_means = k_means.learn_one(x)
    ...     y_pred = k_means.predict_one(x)
    ...     metric = metric.update(k_means.centers, x, y_pred)

    >>> metric
    Cohesion: 1.682748

    """

    @property
    def bigger_is_better(self):
        return False

    def _eval(self, centers, point, y_pred):
        return math.sqrt(utils.math.minkowski_distance(centers[y_pred], point, 2))


class DaviesBouldin(base_internal_clustering.InternalClusteringMetrics):
    """Davies-Bouldin index (DB).

    The Davies-Bouldin index (DB) is an old but still widely used inernal validaion measure.
    DB uses intra-cluster variance and inter-cluster center disance to find the worst partner
    cluster, i.e., the closest most scattered one for each cluster. Thus, minimizing DB gives
    us the optimal number of clusters.

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
    >>> metric = metrics.DaviesBouldin()

    >>> for x, _ in stream.iter_array(X):
    ...     k_means = k_means.learn_one(x)
    ...     y_pred = k_means.predict_one(x)
    ...     metric = metric.update(k_means.centers, x, y_pred)

    >>> metric
    DaviesBouldin: 0.22583

    """

    def __init__(self):
        super().__init__()
        self._inter_cluster_distances = {}
        self._n_points_by_clusters = {}
        self._total_points = 0
        self._centers = {}
        self.sample_correction = {}

    def update(self, centers, point, y_pred, sample_weight=1.0):

        distance = math.sqrt(utils.math.minkowski_distance(centers[y_pred], point, 2))

        # To trace back
        self.sample_correction = {"distance": distance}

        if y_pred not in self._inter_cluster_distances.keys():
            self._inter_cluster_distances[y_pred] = distance
            self._n_points_by_clusters[y_pred] = 1
        else:
            self._inter_cluster_distances[y_pred] += distance
            self._n_points_by_clusters[y_pred] += 1

        self._centers = centers

        return self

    def revert(self, centers, point, y_pred, sample_weight=1.0, correction=None):

        self._inter_cluster_distances[y_pred] -= correction["distance"]
        self._n_points_by_clusters[y_pred] -= 1
        self._centers = centers

        return self

    def get(self):
        max_partner_clusters_index = -math.inf
        n_clusters = len(self._inter_cluster_distances)
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                distance_ij = math.sqrt(
                    utils.math.minkowski_distance(self._centers[i], self._centers[j], 2)
                )
                ij_partner_cluster_index = (
                    self._inter_cluster_distances[i] / self._n_points_by_clusters[i]
                    + self._inter_cluster_distances[j] / self._n_points_by_clusters[j]
                ) / distance_ij
                if ij_partner_cluster_index > max_partner_clusters_index:
                    max_partner_clusters_index = ij_partner_cluster_index
        return max_partner_clusters_index / n_clusters

    @property
    def bigger_is_better(self):
        return False


class IIndex(base_internal_clustering.InternalClusteringMetrics):
    """I-Index (I).

    I-Index (I) adopts the maximum distance between cluster centers. It also shares the type of
    formulation numerator-separation/denominator-compactness. For compactness, the distance from
    a data point to its cluster center is also used like CH.

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
    >>> metric = metrics.DaviesBouldin()

    >>> for x, _ in stream.iter_array(X):
    ...     k_means = k_means.learn_one(x)
    ...     y_pred = k_means.predict_one(x)
    ...     metric = metric.update(k_means.centers, x, y_pred)

    >>> metric
    IIndex: 6.836566

    """

    def __init__(self):
        super().__init__()
        self._center_all_points = {}
        self._ssq_points_cluster_centers = 0
        self._ssq_points_center = 0
        self._furthest_cluster_distance = 0
        self._n_clusters = 0
        self._dim = 0
        self.sample_correction = {}
        self._initialized = False

    @staticmethod
    def _find_furthest_cluster_distance(centers):
        n_centers = len(centers)
        max_distance = -math.inf
        for i in range(n_centers):
            for j in range(i + 1, n_centers):
                distance_ij = math.sqrt(
                    utils.math.minkowski_distance(centers[i], centers[j], 2)
                )
                if distance_ij > max_distance:
                    max_distance = distance_ij
        return max_distance

    def update(self, centers, point, y_pred, sample_weight=1.0):

        self._furthest_cluster_distance = self._find_furthest_cluster_distance(centers)

        if not self._initialized:
            self._center_all_points = {i: stats.Mean() for i in point}
            self._dim = len(point)
            self._initialized = True

        for i in self._center_all_points:
            self._center_all_points[i].update(point[i], w=sample_weight)
        center_all_points = {
            i: self._center_all_points[i].get() for i in self._center_all_points
        }

        distance_point_cluster_center = math.sqrt(
            utils.math.minkowski_distance(centers[y_pred], point, 2)
        )
        distance_point_center = math.sqrt(
            utils.math.minkowski_distance(center_all_points, point, 2)
        )
        self._ssq_points_cluster_centers += distance_point_cluster_center
        self._ssq_points_center += distance_point_center
        self._n_clusters = len(centers)

        # To trace back
        self.sample_correction = {
            "distance_point_cluster_center": distance_point_cluster_center,
            "distance_point_center": distance_point_center,
        }

        return self

    def revert(self, centers, point, y_pred, sample_weight=1.0, correction=None):

        self._furthest_cluster_distance = self._find_furthest_cluster_distance(centers)

        for i in self._center_all_points:
            self._center_all_points[i].update(point[i], w=-sample_weight)
        center_all_points = {
            i: self._center_all_points[i].get() for i in self._center_all_points
        }

        self._ssq_points_cluster_centers -= correction["distance_point_cluster_center"]
        self._ssq_points_center -= correction["distance_point_center"]
        self._n_clusters = len(centers)
        self._dim = len(point)

        return self

    def get(self):
        return (
            1
            / self._n_clusters
            * self._ssq_points_center
            / self._ssq_points_cluster_centers
            * self._furthest_cluster_distance
        ) ** self._dim

    @property
    def bigger_is_better(self):
        return False


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
    ...     [4, 0],
    ...     [-2, 2],
    ...     [-2, 4],
    ...     [-2, 0]
    ... ]

    >>> k_means = cluster.KMeans(n_clusters=3, halflife=0.4, sigma=3, seed=0)
    >>> metric = metrics.Separation()

    >>> for x, _ in stream.iter_array(X):
    ...     k_means = k_means.learn_one(x)
    ...     y_pred = k_means.predict_one(x)
    ...     metric = metric.update(k_means.centers, x, y_pred)

    >>> metric
    Separation: 4.54563

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
    >>> metric = metrics.XieBeni()

    >>> for x, _ in stream.iter_array(X):
    ...     k_means = k_means.learn_one(x)
    ...     y_pred = k_means.predict_one(x)
    ...     metric = metric.update(k_means.centers, x, y_pred)

    >>> metric
    XieBeni: 0.397043

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
        n_centers = max(centers) + 1
        for i in range(n_centers):
            for j in range(i + 1, n_centers):
                separation_ij = utils.math.minkowski_distance(centers[i], centers[j], 2)
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
        self._minimum_separation = minimum_separation

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
