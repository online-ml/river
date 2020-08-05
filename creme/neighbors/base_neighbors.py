from collections import deque

import numpy as np
from sklearn.neighbors import KDTree


from creme.utils.skmultiflow_utils import get_dimensions


class SlidingWindow:
    """Keep a fixed-size sliding window of the most recent data samples.

    Parameters
    ----------
    window_size : int, optional (default=1000)
        The window's size.

    Raises
    ------
    ValueError
        If at any moment, a sample with a different number of attributes than
         those already observed is passed.

    Notes
    -----
    It updates its stored samples by the FIFO method, which means
    that when size limit is reached, old samples are dumped to give
    place to new samples.

    The internal buffer does not keep order of the stored samples,
    when the size limit is reached, the older samples are overwritten
    with new ones (circular buffer).

    """

    def __init__(self, window_size=1000):
        self.window_size = window_size
        self._n_features = -1
        self._n_targets = -1
        self._X_queue = None
        self._y_queue = None
        self._is_initialized = False

    def _configure(self):
        self._X_queue = np.zeros((0, self._n_features))
        self._y_queue = deque()
        self._is_initialized = True

    def add_sample(self, X, y):
        """Add a (single) sample to the sample window.

        X : numpy.ndarray of shape (1, n_features)
            1D-array of feature for a single sample.

        y : numpy.ndarray of shape (1, n_targets)
            1D-array of targets for a single sample.

        Raises
        ------
        ValueError: If at any moment, a sample with a different number of
        attributes than that of the n_attributes parameter is passed, a
        ValueError is raised.

        TypeError: If the buffer type is altered by the user, or is not
        correctly initialized, a TypeError may be raised.

        """
        if not self._is_initialized:
            self._n_features = get_dimensions(X)[1]
            self._n_targets = get_dimensions(y)[1]
            self._configure()

        if self._n_features != get_dimensions(X)[1]:
            raise ValueError("Inconsistent number of features in X: {}, previously observed {}.".
                             format(get_dimensions(X)[1], self._n_features))

        if self.size == self.window_size:
            # Delete oldest sample
            self._X_queue = np.delete(self._X_queue, 0, axis=0)
            self._y_queue.popleft()

        self._X_queue = np.vstack((self._X_queue, X))
        self._y_queue = self._y_queue.append(y)

    def delete_oldest_sample(self):
        """Delete the oldest sample in the window. """
        if self.size > 0:
            self._X_queue = self._X_queue[1:, :]
            self._y_queue.popleft()

    def reset(self):
        """Reset the sliding window. """
        self._n_features = -1
        self._n_targets = -1
        self._X_queue = None
        self._y_queue = None
        self._is_initialized = False

    @property
    def features_buffer(self):
        """Get the features buffer.

        The shape of the buffer is (window_size, n_features).
        """
        return self._X_queue

    @property
    def targets_buffer(self):
        """Get the targets buffer

        The shape of the buffer is (window_size, n_targets).
        """
        return self._y_queue

    @property
    def n_targets(self):
        """Get the number of targets. """
        return self._n_targets

    @property
    def n_features(self):
        """Get the number of features. """
        return self._n_features

    @property
    def size(self):
        """Get the window size. """
        return 0 if self._X_queue is None else self._X_queue.shape[0]


class BaseNeighbors:
    """Base class for neighbors-based estimators. """
    def __init__(self, n_neighbors=5, max_window_size=1000, leaf_size=30, metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.max_window_size = max_window_size
        self.leaf_size = leaf_size
        if metric not in self.valid_metrics():
            raise ValueError('Invalid metric: {}.\n'
                             'Valid options are: {}'.format(metric, self.valid_metrics()))
        self.metric = metric
        self.data_window = SlidingWindow(window_size=max_window_size)

    def _get_neighbors(self, X):
        tree = KDTree(self.data_window.features_buffer, self.leaf_size, metric=self.metric)
        dist, idx = tree.query(X=X, k=self.n_neighbors)
        return dist, idx

    def reset(self):
        """Reset estimator. """
        self.data_window.reset()
        return self

    @staticmethod
    def valid_metrics():
        """Get valid distance metrics for the KDTree. """
        return KDTree.valid_metrics
