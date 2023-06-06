from __future__ import annotations

import collections
import functools
import operator
import typing

from river import utils

from .base import BaseNN, DistanceFunc, FunctionWrapper


class LazySearch(BaseNN):
    """Exact nearest neighbors using a lazy search estrategy.

    Parameters
    ----------
    window_size
        Size of the sliding window use to search neighbors with.
    min_distance_keep
        The minimum distance (similarity) to consider adding a point to the window.
        E.g., a value of 0.0 will add even exact duplicates.
    dist_func
        A distance function which accepts two input items to compare. If not set,
        use the Minkowski distance with `p=2`.

    Notes
    -----
    Updates are by default stored by the FIFO (first in first out) method,
    which means that when the size limit is reached, old samples are dumped to
    give room for new samples. This is circular, meaning that older points
    are dumped first. This also gives the implementation a temporal aspect,
    because older samples are replaced with newer ones.

    The parameter `min_dinstance_keep` controls the addition of new items to the
    window - items that are far enough away (> min_distance_keep) are added to
    the window. Thus a value of 0 indicates that we add all points, and
    increasing from 0 makes it less likely we will keep a new item.

    """

    def __init__(
        self,
        window_size: int = 50,
        min_distance_keep: float = 0.0,
        dist_func: DistanceFunc | FunctionWrapper | None = None,
    ):
        self.window_size = window_size

        # A minimum distance (similarity) to determine adding to window
        # The model will perform better with a more diverse window
        # Since the distance function can be anything, it could be < 0
        self.min_distance_keep = min_distance_keep

        if dist_func is None:
            dist_func = functools.partial(utils.math.minkowski_distance, p=2)
        self.dist_func = dist_func

        self.window: collections.deque = collections.deque(maxlen=self.window_size)

    def append(self, item: typing.Any, extra: typing.Any | None = None, **kwargs):
        """Add a point to the window, optionally with extra metadata.

        Parameters
        ----------
        item
            The data intended to be provided to the distance function. It is always
            the first item in the window, and typically this will be a tuple
            (x,y) with features `x` and class or value `y`.
        extra:
            An extra set of metadata to add to the window that is not passed to
            the distance function, and allows easy customization without needing
            to always write a custom distance function.

        """
        self.window.append((item, *(extra or [])))

    def update(
        self,
        item: typing.Any,
        n_neighbors: int = 1,
        extra: typing.Any | None = None,
    ):
        """Update the window with a new point, only added if > min distance.

        If min distance is 0, we do not need to do the calculation. The item
        (and extra metadata) will not be added to the window if it is too close
        to an existing point.

        Parameters
        ----------
        item
            The data intended to be provided to the distance function.
        extra
            Metadata that is separate from the item that should also be added
            to the window, but is not included to be passed to the distance
            function.

        Returns
        -------
        A boolean (true/false) to indicate if the point was added.

        """
        # If min distance is 0, we add all points
        if self.min_distance_keep == 0:
            self.append(item, extra=extra)
            return True

        # Don't add VERY similar points to window
        nearest = self.search(item, n_neighbors)

        # Distance always in the last index, (item <extra> distance)
        if not nearest or nearest[0][-1] < self.min_distance_keep:
            self.append(item, extra=extra)
            return True
        return False

    def search(self, item: typing.Any, n_neighbors: int, **kwargs):
        """Find the `n_neighbors` closest points to `item`, along with their distances."""
        # Compute the distances to each point in the window
        # The window is (item, <extra>, distance)
        points = ((*p, self.dist_func(item, p[0])) for p in self.window)

        # Return the k closest points
        return tuple(map(list, zip(*sorted(points, key=operator.itemgetter(-1))[:n_neighbors])))
