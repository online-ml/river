import collections
import functools
import operator
from typing import Any, Callable, Tuple

from river import utils

__all__ = ["NearestNeighbors", "MinkowskiNeighbors"]

DistanceFunc = Callable[[Any, Any], float]


class NearestNeighbors:
    """A basic data structure to hold nearest neighbors.

    Parameters
    ----------

    n_neighbors
        Number of neighbors to use.

    window_size
        Size of the sliding window use to search neighbors with.

    min_distance_keep
        The minimum distance (similarity) to consider adding a point to the window.
        E.g., a value of 0.0 will add even exact duplicates. Default is 0.05 to add
        similar but not exactly the same points.

    distance_func
        An required distance function that accept two input items to compare
        and optional parameters. It's recommended to use functools.partial.

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
        n_neighbors: int = 5,
        window_size: int = 1000,
        min_distance_keep: float = 0.0,
        distance_func: DistanceFunc = None,
    ):
        self.n_neighbors = n_neighbors
        self.window_size = window_size

        # A minimum distance (similarity) to determine adding to window
        # The model will perform better with a more diverse window
        # Since the distance function can be anything, it could be < 0
        self.min_distance_keep = min_distance_keep

        self.distance_func = distance_func
        self.reset()

    def append(self, item: Any, extra: [Tuple, list] = None):
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

    def update(self, item: Any, n_neighbors=1, extra: [Tuple, list] = None):
        """Update the window with a new point, only added if > min distance.

        If min distance is 0, we do not need to do the calculation. The item
        (and extra metadata) will not be added to the window if it is too close
        to an existing point.

        Parameters
        ----------
        item
            The data intended to be provided to the distance function. For a
            standard case, it is expected to be a tuple with x first and y
            second.
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
        nearest = self.find_nearest(item, n_neighbors)

        # Distance always the last index, (x,y <extra> distance)
        if not nearest or nearest[0][-1] < self.min_distance_keep:
            self.append(item, extra=extra)
            return True
        return False

    def find_nearest(self, item: Any, n_neighbors=1):
        """Find the `n_neighbors` closest points to `x`, along with their distances.

        This function assumes the x is a tuple or list with x[0] having relevant
        data for the distance calculation.

        """
        # Compute the distances to each point in the window
        # Item is JUST the (x,y) however the window is (item, <extra>, distance)
        points = ((*p, self.distance_func(item, p[0])) for p in self.window)

        # Return the k closest points (last index is distance)
        return sorted(points, key=operator.itemgetter(-1))[:n_neighbors]

    def reset(self) -> "NearestNeighbors":
        """Reset window"""
        self.window = collections.deque(maxlen=self.window_size)


def custom_minkowski(a, b, p):
    """Custom minkoski function. Must be global to be pickle-able."""
    return utils.math.minkowski_distance(a[0], b[0], p=p)


class MinkowskiNeighbors(NearestNeighbors):
    """NearestNeighbors using the Minkowski metric as the distance with p=2.

    You can still overwrite the distance_func here, however the default is
    provided for the nearest neighbors classifiers to use, expecting that a
    typical user will not provide a custom function.

    Parameters
    ----------

    n_neighbors
        Number of neighbors to use.

    window_size
        Size of the sliding window use to search neighbors with.

    min_distance_keep
        The minimum distance (similarity) to consider adding a point to the window.
        E.g., a value of 0.0 will add even exact duplicates. Default is 0.05 to add
        similar but not exactly the same points.

    distance_func
        An optional distance function that should accept an a=, b=, and any
        custom set of kwargs (defined in distance_func_kwargs). If not defined,
        the default Minkowski distance is used.

    p
        p-norm value for the Minkowski metric. When `p=1`, this corresponds to the
        Manhattan distance, while `p=2` corresponds to the Euclidean distance.
        Valid values are in the interval $[1, +\\infty)$
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        window_size: int = 1000,
        min_distance_keep: float = 0.0,
        distance_func: DistanceFunc = None,
        p: float = 2.0,
    ):

        self.p = p
        super().__init__(
            n_neighbors=n_neighbors,
            window_size=window_size,
            distance_func=distance_func
            or functools.partial(custom_minkowski, p=self.p),
            min_distance_keep=min_distance_keep,
        )
