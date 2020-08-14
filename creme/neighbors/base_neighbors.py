import itertools

import numpy as np
from scipy.spatial import cKDTree

from creme.utils.skmultiflow_utils import get_dimensions


class KNeighborsBuffer:
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
        self._size = 0
        self._next_insert = 0
        self._imask = None
        self._X = None
        self._y = None
        self._is_initialized = False

    def _configure(self):
        # Binary instance mask to filter data in the buffer
        self._imask = np.zeros(self.window_size, dtype=bool)
        self._X = np.zeros((self.window_size, self._n_features))
        self._y = [None for _ in range(self.window_size)]
        self._is_initialized = True

    def reset(self):
        """Reset the sliding window. """
        self._n_features = -1
        self._n_targets = -1
        self._size = 0
        self._next_insert = 0
        self._imask = None
        self._X = None
        self._y = None
        self._is_initialized = False

        return self

    def add_one(self, x, y):
        """Add a (single) sample to the sample window.

        x : numpy.ndarray of shape (1, n_features)
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
            self._n_features = get_dimensions(x)[1]
            self._n_targets = get_dimensions(y)[1]
            self._configure()

        if self._n_features != get_dimensions(x)[1]:
            raise ValueError("Inconsistent number of features in X: {}, previously observed {}.".
                             format(get_dimensions(x)[1], self._n_features))

        # if self.size == self.window_size:
        #     self.remove_one()

        self._X[self._next_insert, :] = x
        self._y[self._next_insert] = y

        # Update the instance storing logic
        self._imask[self._next_insert] = True  # Mark slot as filled
        self._next_insert = self._next_insert + 1 if self._next_insert < self.window_size - 1 \
            else 0
        self._size += 1

    def remove_one(self):
        """Delete the oldest sample in the window. """
        if self.size > 0:
            self._next_insert = self._next_insert - 1 if self._next_insert > 0 else \
                self.window_size - 1
            self._imask[self._next_insert] = False  # Mark slot as free
            self._size -= 1

    def clear(self):
        """Clear all stored elements."""
        self._next_insert = 0
        self._size = 0
        # Just reset the instance filtering mask, not the X buffer
        self._imask = np.zeros(self.window_size, dtype=bool)

    @property
    def features_buffer(self):
        """Get the features buffer.

        The shape of the buffer is (window_size, n_features).
        """
        return self._X[self._imask, :]  # Only return the actually filled instances

    @property
    def targets_buffer(self):
        """Get the targets buffer

        The shape of the buffer is (window_size, n_targets).
        """
        return list(itertools.compress(self._y, self._imask))

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
        return self._size


class BaseNeighbors:
    """Base class for neighbors-based estimators. """
    def __init__(self, n_neighbors=5, max_window_size=1000, leaf_size=30, p=2):
        self.n_neighbors = n_neighbors
        self.max_window_size = max_window_size
        self.leaf_size = leaf_size
        if p < 1:
            raise ValueError('Invalid Minkowski p-norm value: {}.\n'
                             'Values must be greater than or equal to 1'.format(p))
        self.p = p
        self.data_window = KNeighborsBuffer(window_size=max_window_size)

    def _get_neighbors(self, X):
        tree = cKDTree(self.data_window.features_buffer, leafsize=self.leaf_size)
        dist, idx = tree.query(X.reshape(1, -1), k=self.n_neighbors, p=self.p)
        return dist, idx

    def reset(self):
        """Reset estimator. """
        self.data_window.reset()
        return self
