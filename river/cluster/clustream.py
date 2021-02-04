import math
import sys

from river import base, cluster
from river.cluster.clustream_kernel import ClustreamKernel


class Clustream(base.Clusterer):
    """Clustream

    It maintains statistical information about the data using micro-clusters.
    These micro-clusters are temporal extensions of cluster feature vectors.
    The micro-clusters are stored at snapshots in time following a pyramidal
    pattern. This pattern allows to recall summary statistics from different
    time horizons in [1]_.

    Parameters
    ----------
    seed
       If int, seed is the seed used by the random number generator;
       If RandomState instance, seed is the random number generator;
       If None, the random number generator is the RandomState instance used
       by `np.random`.
       It is used in the `KMeans` algorithm for the offline clustering, which is the native `KMneans` implemented in `River`.

    time_window
      The rang of the window
      if the current time is T and the time window is h, we should only consider
      about the data that arrived within the period (T-h,T)

    max_kernels
      The Maximum number of micro kernels to use

    kernel_radius_factor
      Multiplier for the kernel radius
      When deciding to add a new data point to a micro-cluster, the maximum boundary
      is defined as a factor of the kernel_radius_factor of the RMS deviation of the
      data points in the micro-cluster from the centroid

    number_of_clusters
        the clusters returned by the Kmeans algorithm using the summaries statistics

    kwargs
        Other parameters passed to the incremental kmeans at `river.cluster.KMeans`

    Attributes
    ----------
    centers : dict
        Central positions of each cluster.

    Examples
    --------

    In the following implementation, due to the limited number of points passed through the model,
    max_kernels and time_window are set relatively low. Moreover, all points are learnt before any predictions are made.
    The halflife are set at 0.4, passing from cluster.KMeans, to test the functionality of keyword arguments.

    >>> from river import cluster
    >>> from river import stream

    >>> X = [
    ...     [1, 2],
    ...     [1, 4],
    ...     [1, 0],
    ...     [4, 2],
    ...     [4, 4],
    ...     [4, 0]
    ... ]

    >>> clustream_test = cluster.Clustream(time_window = 1, max_kernels = 3, number_of_clusters = 2, seed = 0, halflife = 0.4)

    >>> for i, (x, _) in enumerate(stream.iter_array(X)):
    ...     clustream_test = clustream_test.learn_one(x)

    >>> clustream_test.predict_one({0: 1, 1: 1})
    1

    >>> clustream_test.predict_one({0: 4, 1: 3})
    0

    References
    ----------
    .. [1] A. Kumar, A. Singh, and R. Singh, 2017. An efficient hybrid-clustream algorithm
       for stream mining. In Proceedings of the 13th International Conference on Signal-Image Technology & Internet-Based System (SITIS).
       DOI: 10.1109/SITIS.2017.77

    """

    def __init__(
        self,
        seed: int = None,
        time_window: int = 1000,
        max_kernels: int = 100,
        kernel_radius_factor: int = 2,
        number_of_clusters: int = 5,
        **kwargs
    ):
        super().__init__()
        self.time_window = time_window
        self.time_stamp = -1
        self.kernels = {n: None for n in range(max_kernels)}
        self.initialized = False
        self.buffer = {}
        self.buffer_size = max_kernels
        self.centers = (
            {}
        )  # add this to retrieve centers later for the evaluation of models through
        self.kernel_radius_factor = kernel_radius_factor
        self.max_kernels = max_kernels
        self.number_of_clusters = number_of_clusters
        self._train_weight_seen_by_model = 0.0
        self.seed = seed
        self.kwargs = kwargs

    def learn_one(self, x, sample_weight=None):
        """Incrementally trains the model.

        Tasks performed before training:
        * Verify instance weight. if not provided, uniform weights (1.0) are assumed.
        * Update weight seen by model.

        Training tasks:
        * determinate closest kernel
        * Check whether instance fits into closest Kernel:
            1- if data fits, put into kernel
            2- if data does not fit, we need to free some space to insert a new kernel
            and this can be done in two ways, delete an old kernel or merge two kernels
            which are close to each other

        Parameters
        ----------
        x
            A dictionary of features

        sample_weight
            Instance weights. If not provided, uniform weights are assumed

        Returns
        ----------
        self

        """
        if sample_weight == 0:
            return
        elif sample_weight is None:
            sample_weight = 1.0

        self._train_weight_seen_by_model += sample_weight

        # merge _learn_one into learn_one
        self.time_stamp += 1

        if not self.initialized:
            if len(self.buffer) < self.buffer_size:
                self.buffer[len(self.buffer)] = ClustreamKernel(
                    x=x,
                    sample_weight=sample_weight,
                    timestamp=self.time_stamp,
                    T=self.kernel_radius_factor,
                    M=self.max_kernels,
                )
                return self
            else:
                for i in range(self.buffer_size):
                    self.kernels[i] = ClustreamKernel(
                        x=self.buffer[i].center,
                        sample_weight=1.0,
                        timestamp=self.time_stamp,
                        T=self.kernel_radius_factor,
                        M=self.max_kernels,
                    )
            self.buffer.clear()
            self.initialized = True

            return self

        """determine closest kernel"""
        closest_kernel = None
        min_distance = sys.float_info.max
        for kernel in self.kernels.values():
            distance = self._distance(x, kernel.center)
            if distance < min_distance:
                closest_kernel = kernel
                min_distance = distance

        """check whether the instance fits into closest kernel"""
        radius = 0.0
        if closest_kernel.weight == 1:
            radius = sys.float_info.max
            center = closest_kernel.center
            for kernel in self.kernels.values():
                if kernel == closest_kernel:
                    continue
                distance = self._distance(kernel.center, center)
                radius = min(distance, radius)
        else:
            radius = closest_kernel.radius

        if min_distance < radius:
            closest_kernel.insert(x, sample_weight, self.time_stamp)
            return self

        """Data does not fit, we need to free some space in order to insert a new kernel"""

        threshold = self.time_stamp - self.time_window

        """try to delete old kernel"""
        for i, kernel in self.kernels.items():
            if kernel.relevance_stamp < threshold:
                self.kernels[i] = ClustreamKernel(
                    x=x,
                    sample_weight=sample_weight,
                    timestamp=self.time_stamp,
                    T=self.kernel_radius_factor,
                    M=self.max_kernels,
                )

                return self

        """try to merge closest two kernels"""
        closest_a = 0
        closest_b = 0
        min_distance = sys.float_info.max
        for i, kernel_i in self.kernels.items():
            center_a = kernel_i.center
            for j, kernel_j in self.kernels.items():
                dist = self._distance(center_a, kernel_j.center)
                if dist < min_distance and j > i:
                    min_distance = dist
                    closest_a = i
                    closest_b = j
        self.kernels[closest_a].add(self.kernels[closest_b])
        self.kernels[closest_b] = ClustreamKernel(
            x=x,
            sample_weight=sample_weight,
            timestamp=self.time_stamp,
            T=self.kernel_radius_factor,
            M=self.max_kernels,
        )

        return self

    def get_micro_clustering_result(self):

        if not self.initialized:
            return {}
        res = {
            i: ClustreamKernel(
                cluster=self.kernels[i], T=self.kernel_radius_factor, M=self.max_kernels
            )
            for i in range(len(self.kernels))
        }
        return res

    def learn_predict_one(self, x, sample_weight=None):
        """
        Compute cluster centers and predict cluster index for each sample.

        Convenience method; equivalent to calling learn_one(x) followed by predict_one(x).

        Parameters
        ----------
        x
            Instance attributes

        sample_weight
            Instance weights. If not provided, uniform weights are assumed

        Returns
        -------
        y
            Integer
            Cluster label
        """

        self.learn_one(x, sample_weight)

        y = self.predict_one(x)

        return y

    @staticmethod
    def _get_closest_kernel(x, micro_clusters):
        """

        Parameters
        ----------
        x
            A dictionary with n items (features).
            Instance attributes. Due to the limitations of the dictionary, x on default is considered as one instance

        micro_clusters
            Dictionary like
            Instance weight. If not provided, uniform weights are assumed

        Returns
        -------
        closest_kernel_index : int
            Index of closest kernel to the given instance

        min_distance: float
            distance between closest kernel to the given instance
        """

        min_distance = sys.float_info.max
        closest_kernel_index = -1
        for i, micro_cluster in micro_clusters.items():
            distance = 0
            for j in range(len(x)):
                distance += (micro_cluster.center[j] - x[j]) * (
                    micro_cluster.center[j] - x[j]
                )
            distance = math.sqrt(distance)
            if distance < min_distance:
                min_distance = distance
                closest_kernel_index = i
        return closest_kernel_index, min_distance

    @staticmethod
    def _distance(point_a, point_b):
        distance = 0.0
        for i in range(len(point_a)):
            d = point_a[i] - point_b[i]
            distance += d * d
        return math.sqrt(distance)

    def predict_one(self, x):
        """Predict cluster index for each sample.

        Convenience method; equivalent to calling partial_fit(x) followed by predict(x).

        Parameters
        ----------
        x
            A dictionary with n items
            Instance attributes

        Returns
        -------
        label
            Integer
            Cluster label

        """

        micro_cluster_centers = {
            i: self.get_micro_clustering_result()[i].center
            for i in range(len(self.get_micro_clustering_result()))
        }

        # implementing the incremental KMeans in River to generate clusters from micro cluster centers
        kmeans = cluster.KMeans(
            n_clusters=self.number_of_clusters, seed=self.seed, **self.kwargs
        )
        for center in micro_cluster_centers.values():
            kmeans = kmeans.learn_one(center)

        self.centers = kmeans.centers

        index, _ = self._get_closest_kernel(x, self.get_micro_clustering_result())
        y = kmeans.predict_one(micro_cluster_centers[index])

        return y
