import itertools
import typing

import numpy as np
from scipy.spatial import cKDTree

from river import base
from river.utils.skmultiflow_utils import get_dimensions


class KNeighborsBuffer:
    """Keep a fixed-size sliding window of the most recent data samples.

    Parameters
    ----------
    window_size
        The size of the window.

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

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self._n_features: int = -1
        self._n_targets: int = -1
        self._size: int = 0
        self._next_insert: int = 0
        self._oldest: int = 0
        self._imask: np.ndarray
        self._X: np.ndarray
        self._y: typing.List
        self._is_initialized: bool = False

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
        self._oldest = 0
        self._imask = None
        self._X = None
        self._y = None
        self._is_initialized = False

        return self

    def append(self, x: np.ndarray, y: base.typing.Target) -> "KNeighborsBuffer":
        """Add a (single) sample to the sample window.

        x
            1D-array of feature for a single sample.

        y
            The target data for a single sample.

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
            raise ValueError(
                "Inconsistent number of features in X: {}, previously observed {}.".format(
                    get_dimensions(x)[1], self._n_features
                )
            )

        self._X[self._next_insert, :] = x
        self._y[self._next_insert] = y

        slot_replaced = self._imask[self._next_insert]

        # Update the instance storing logic
        self._imask[self._next_insert] = True  # Mark slot as filled
        self._next_insert = (
            self._next_insert + 1 if self._next_insert < self.window_size - 1 else 0
        )

        if (
            slot_replaced
        ):  # The oldest sample was replaced (complete cycle in the buffer)
            self._oldest = self._next_insert
        else:  # Actual buffer increased
            self._size += 1

        return self

    def pop(self) -> typing.Union[typing.Tuple[np.ndarray, base.typing.Target], None]:
        """Remove and return the most recent element added to the buffer. """
        if self.size > 0:
            self._next_insert = (
                self._next_insert - 1 if self._next_insert > 0 else self.window_size - 1
            )
            x, y = self._X[self._next_insert], self._y[self._next_insert]
            self._imask[self._next_insert] = False  # Mark slot as free
            self._size -= 1

            return x, y
        else:
            return None

    def popleft(
        self,
    ) -> typing.Union[typing.Tuple[np.ndarray, base.typing.Target], None]:
        """Remove and return the oldest element in the buffer. """
        if self.size > 0:
            x, y = self._X[self._oldest], self._y[self._oldest]
            self._imask[self._oldest] = False  # Mark slot as free
            self._oldest = (
                self._oldest + 1 if self._oldest < self.window_size - 1 else 0
            )
            if self._oldest == self._next_insert:
                # Shift circular buffer and make its starting point be the index 0
                self._oldest = self._next_insert = 0
            self._size -= 1

            return x, y

    def clear(self) -> "KNeighborsBuffer":
        """Clear all stored elements."""
        self._next_insert = 0
        self._oldest = 0
        self._size = 0
        # Just reset the instance filtering mask, not the buffers
        self._imask = np.zeros(self.window_size, dtype=bool)

        return self

    @property
    def features_buffer(self) -> np.ndarray:
        """Get the features buffer.

        The shape of the buffer is (window_size, n_features).
        """
        return self._X[self._imask]  # Only return the actually filled instances

    @property
    def targets_buffer(self) -> typing.List:
        """Get the targets buffer

        The shape of the buffer is (window_size, n_targets).
        """
        return list(itertools.compress(self._y, self._imask))

    @property
    def n_targets(self) -> int:
        """Get the number of targets. """
        return self._n_targets

    @property
    def n_features(self) -> int:
        """Get the number of features. """
        return self._n_features

    @property
    def size(self) -> int:
        """Get the window size. """
        return self._size


class BaseNeighbors:
    """Base class for neighbors-based estimators. """

    def __init__(
        self,
        n_neighbors: int = 5,
        window_size: int = 1000,
        leaf_size: int = 30,
        p: float = 2,
        **kwargs
    ):
        self.n_neighbors = n_neighbors
        self.window_size = window_size
        self.leaf_size = leaf_size
        self._kwargs = kwargs

        if p < 1:
            raise ValueError(
                "Invalid Minkowski p-norm value: {}.\n"
                "Values must be greater than or equal to 1".format(p)
            )
        self.p = p
        self.data_window = KNeighborsBuffer(window_size=window_size)

    def _get_neighbors(self, x):
        X = self.data_window.features_buffer
        tree = cKDTree(X, leafsize=self.leaf_size, **self._kwargs)
        dist, idx = tree.query(x.reshape(1, -1), k=self.n_neighbors, p=self.p)

        # We make sure dist and idx is 2D since when k = 1 dist is one dimensional.
        if not isinstance(dist[0], np.ndarray):
            dist = [dist]
            idx = [idx]
        return dist, idx

    def reset(self) -> "BaseNeighbors":
        """Reset estimator. """
        self.data_window.reset()

        return self
