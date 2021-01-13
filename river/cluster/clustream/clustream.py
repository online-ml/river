from river import base
import math
from river.cluster.clustream.clustream_kernel import ClustreamKernel
import sys
from sklearn.cluster import KMeans
import numpy as np

class Clustream(base.Clusterer):
    """Clustream

    It maintains statistical information about the data using micro-clusters.
    These micro-clusters are temporal extensions of cluster feature vectors.
    The micro-clusters are stored at snapshots in time following a pyramidal
    pattern. This pattern allows to recall summary statistics from different
    time horizons in [1]_.

    Parameters
    ----------
    random_state: int, RandomState instance or None, optional (default=None)
       If int, random_state is the seed used by the random number generator;
       If RandomState instance, random_state is the random number generator;
       If None, the random number generator is the RandomState instance used
       by `np.random`.
       It is used in the kmeans algorithm for the offline clustering

    time_window: int (Default : 1000)
      The rang of the window
      if the current time is T and the time window is h, we should only consider
      about the data that arrived within the period (T-h,T)

    max_kernels: int (Default: 100)
      The Maximum number of micro kernels to use

    kernel_radius_factor: int (Default: 2)
      Multiplier for the kernel radius
      When deciding to add a new data point to a micro-cluster, the maximum boundary
      is defined as a factor of the kernel_radius_factor of the RMS deviation of the
      data points in the micro-cluster from the centroid

    number_of_clusters: int (Default : 5)
        the clusters returned by the Kmeans algorithm using the summaries statistics

    References
    ----------
    .. [1] A. Kumar , A. Singh, and R. Singh. An efficient hybrid-clustream algorithm
       for stream mining

    """
    def __init__(self, random_state=None,
                 time_window=1000,
                 max_kernels=100,
                 kernel_radius_factor=2,
                 number_of_clusters=5):
        super().__init__()
        self.time_window = time_window
        self.time_stamp = -1
        self.kernels = {n: None for n in range(max_kernels)}
        self.initialized = False
        self.buffer = {}
        self.buffer_size = max_kernels
        self.T = kernel_radius_factor
        self.M = max_kernels
        self.k = number_of_clusters
        self._train_weight_seen_by_model = 0.0
        self.random_state = random_state

    def learn_one(self, x, weight=None):
        """Incrementally trains the model. Train samples (instances) are composed of x attributes.

        Tasks performed before training:
        * Verify instance weight. if not provided, uniform weights (1.0) are assumed.
        * If more than one instance is passed, loop through X and pass instances one at a time.
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

        """
        if weight is None:
            weight = 1.0
        if weight != 0.0:
            self._train_weight_seen_by_model = weight
            self._learn_one(x, weight)

    def _learn_one(self, x, weight):

        self.time_stamp += 1

        if not self.initialized:
            if len(self.buffer) < self.buffer_size:
                self.buffer.append(ClustreamKernel(x=x, weight=weight, dimensions=1, timestamp=self.time_stamp, T=self.T, M=self.M))
                return
            else:
                for i in range(self.buffer_size):
                    self.kernels[i] = ClustreamKernel(x=self.buffer[i].get_center(), weight=1.0, dimensions=1,
                                                      timestamp=self.time_stamp, T=self.T, M=self.M)
            self.buffer.clear()
            self.initialized = True

            return


        """determine closest kernel"""
        closest_kernel = None
        min_distance = sys.float_info.max
        for i in range(len(self.kernels)):
            distance = self._distance(x, self.kernels[i].get_center())
            if distance < min_distance:
                closest_kernel = self.kernels[i]
                min_distance = distance

        """check whether the instance fits into closest kernel"""
        radius = 0.0
        if closest_kernel.get_weight() == 1:
            radius = sys.float_info.max
            center = closest_kernel.get_center()
            for i in range(len(self.kernels)):
                if self.kernels[i] == closest_kernel:
                    continue
                distance = self._distance(self.kernels[i].get_center(), center)
                radius = min(distance, radius)
        else:
            radius = closest_kernel.get_radius()

        if min_distance < radius:
            closest_kernel.insert(x, weight, self.time_stamp)

        """Data does not fit, we need to free some space in order to insert a new kernel"""

        threshold = self.time_stamp - self.time_window

        """try to delete old kernel"""
        for i in range(len(self.kernels)):
            if self.kernels[i].get_relevance_stamp() < threshold:

                self.kernels[i] = ClustreamKernel(X=X, weight=weight, dimensions=dim, timestamp=self.time_stamp,
                                                  T=self.T, M=self.M)

                return

        """try to merge closest two kernels"""
        closest_a = 0
        closest_b = 0
        for i in range(len(self.kernels)):
            center_a = self.kernels[i].get_centers()
            for j in range(i+1, len(self.kernels())):
                dist = self._distance(center_a, self.kernels[j].get_center())
                if dist < min_distance:
                    min_distance = dist
                    closest_a = i
                    closest_b = j
        assert closest_a != closest_b
        self.kernels[closest_a].add(self.kernels[closest_b])
        self.kernels[closest_b] = ClustreamKernel(x=x, weight=weight, dimensions=dim, timestamp=self.time_stamp,
                                                  T=self.T, M=self.M)

    def get_micro_clustering_result(self):

        if not self.initialized:
            return {}
        res = {i: None for i in range(len(self.kernels))}
        for i in range(len(res)):
            res[i] = ClustreamKernel(cluster=self.kernels[i], T=self.T, M=self.M)
        return res

    def get_clustering_result(self):

        if not self.initialized:
            return {}
        micro_cluster_centers = {}
        for i in len(self.get_micro_clustering_results()):
            micro_cluster_centers[i] = self.get_micro_clustering_results()[i].get_center()

        # turn micro_cluster_centers to numpy array to use kmeans
        micro_cluster_centers_np = []
        for key in micro_cluster_centers.keys():
            micro_cluster_centers_np.append(list(micro_cluster_centers[key].values()))
        micro_cluster_centers_np = np.array(micro_cluster_centers_np)

        # apply kmeans on np.array
        kmeans = KMeans(n_clusters=self.k, random_state=self.random_state).fit(micro_cluster_centers_np)

        return dict(enumerate(kmeans.flatten(), 1)) # convert an numpy array to a dictionary

    def learn_predict_one(self, x, weight = None):
        """
        Compute cluster centers and predict cluster index for each sample.

        Convenience method; equivalent to calling learn_one(x) followed by predict_one(x).

        Parameters
        ----------
        x
            A dictionary with n items
            Instance attributes

        weight
            Float or dictionary-like
            Instance weights. If not provided, uniform weights are assumed

        Returns
        -------
        index
            Integer
            Cluster label
        """
        if weight is None:
            weight = 1.0
        if weight != 0.0:
            self._train_weight_seen_by_model += weight
            self._partial_fit(x, weight)

        micro_cluster_centers = {}
        for i in len(self.get_micro_clustering_results()):
            micro_cluster_centers[i] = self.get_micro_clustering_results()[i].get_center()

        # turn micro_cluster_centers to numpy array to use kmeans
        micro_cluster_centers_np = []
        for key in micro_cluster_centers.keys():
            micro_cluster_centers_np.append(list(micro_cluster_centers[key].values()))
        micro_cluster_centers_np = np.array(micro_cluster_centers_np)

        # apply kmeans on np.array
        kmeans = KMeans(n_clusters=self.k, random_state=self.random_state).fit(micro_cluster_centers_np)

        index, _ = self._get_closest_kernel(x, micro_cluster_centers)
        y = kmeans.labels_[index]

        return y

    @staticmethod
    def _get_closest_kernel(selƒ, x, micro_clusters):
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
        closest_kernel = None
        closest_kernel_index = -1
        for i, micro_cluster in micro_clusters.items():
            distance = self._distance(micro_cluster.ger_center(), x)
            if distance < min_distance:
                min_distance = distance
                closest_kernel = micro_cluster
                closest_kernel_index = i
        return closest_kernel_index, min_distance

    @staticmethod
    def implements_micro_clustering():
        return True

    def get_name(self):
        return "clustream" + str(self.time_window)

    @staticmethod
    def _distance(point_a, point_b):
        distance = 0.0
        for i in range(len(point_a)):
            d = point_a[i] - point_b[i]
            distance += d * d
        return math.sqrt(distance)

    def predict_one(self, x):
        """
        predict cluster index for each sample.

        Convenience method; equivalent to calling partial_fit(X) followed by predict(X).

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

        micro_cluster_centers = {}
        for i in len(self.get_micro_clustering_results()):
            micro_cluster_centers[i] = self.get_micro_clustering_results()[i].get_center()

        # turn micro_cluster_centers to numpy array to use kmeans
        micro_cluster_centers_np = []
        for key in micro_cluster_centers.keys():
            micro_cluster_centers_np.append(list(micro_cluster_centers[key].values()))
        micro_cluster_centers_np = np.array(micro_cluster_centers_np)

        # apply kmeans on np.array
        kmeans = KMeans(n_clusters=self.k, random_state=self.random_state).fit(micro_cluster_centers_np)

        index, _ = self._get_closest_kernel(x, micro_cluster_centers)
        y = kmeans.labels_[index]

        return y

    def predict_proba(self, x):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def score(self, x, y):
        raise NotImplementedError

    def fit(self, x, y, classes = None, weight = None):
        raise NotImplementedError

    def get_info(self):
        raise NotImplementedError