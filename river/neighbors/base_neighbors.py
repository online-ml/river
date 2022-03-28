import collections
import operator
from typing import Callable, Tuple

from river import base, utils

__all__ = ["NearestNeighbors"]


class NearestNeighbors(base.Classifier):
    """
    K-Nearest Neighbors (KNN) to get neighbors back. It is also a base
    class for the other family of neighbors. It extends the base.Classifier
    but allows for multiclass prediction.

    This works by storing a buffer with the `window_size` most recent observations.
    A brute-force search is used to find the `n_neighbors` nearest observations
    in the buffer to make a prediction.

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

    distance_func_kwargs
        A dictionary to pass as kwargs to your distance function, in addition
        to a= and b=. If distance_func is set to None, these are ignored,
        and are set to including the power parameter for the Minkowski metric.
        For this parameter, when `p=1`, this corresponds to the Manhattan
        distance, while `p=2` corresponds to the Euclidean distance.

    Examples
    --------
    >>> from river import datasets, neighbors
    >>> from river import evaluate, metrics
    >>> import uuid
    >>> dataset = datasets.Phishing()

    >>> model = neighbors.NearestNeighbors(window_size=50)
    >>> for x, y in dataset.take(10):
    ...     model = model.learn_one(x, y)

    >>> # Add an extra vector of metadata to get back (e.g. a UID)
    >>> # If you have named features, use a dict in the list or tuple
    >>> model = neighbors.NearestNeighbors(window_size=50)
    >>> for x, y in dataset.take(10):
    ...     model = model.learn_one(x, y, extra=[str(uuid.uuid4())])

    Notes
    -----
    Updates are by default stored by the FIFO (first in first out) method,
    which means that when the size limit is reached, old samples are dumped to
    give room for new samples. This is circular, meaning that older points
    are dumped first. This also gives the implementation a temporal aspect,
    because older samples are replaced with newer ones. However, if
    min_distance_keep is set to be != 0 every sample given to learn may not be
    added to the window. When using this class, the "x" array for the window
    can be a set of any (x, y, <attributes>) and the class always assumes x[0]
    to be features x, x[1] to be y, and x[2] .. x[n-2] where N is the length of
    the array to be any set of custom features added by the class. After the
    metric is run, the distance should always be last in the array, x[-1]
    or x[n-1] given an array of length n.
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        window_size: int = 1000,
        min_distance_keep: float = 0.0,
        distance_func: Callable = None,
        distance_func_kwargs: dict = None,
    ):
        self.n_neighbors = n_neighbors
        self.window_size = window_size

        # If we don't have a distance func, set to use minkowski distance
        # We have functions for this to support updating it after model init
        if not distance_func:
            self.set_default_distance_func(distance_func_kwargs)
        else:
            self.set_distance_func(distance_func, distance_func_kwargs)

        # Distance must be 0 / positive integer
        if min_distance_keep < 0:
            raise ValueError(
                "Invalid min_distance_keep: {}.\n"
                "Values must be greater than or equal to 0".format(min_distance_keep)
            )

        # A minimum distance (similarity) to determine adding to window
        # The model will perform better with a more diverse window
        self.min_distance_keep = min_distance_keep
        self.reset()

    @property
    def _multiclass(self):
        return True

    def learn_one(self, x: dict, y=None, extra: [Tuple, list] = None):
        """Learn a set of features `x` and optional class `y`.
        Parameters:
            x: A dictionary of features.
            y: A class (optional if known).
            extra: an optional list or tuple of features to store
        Returns:
            self
        """
        self.update(x, y, n_neighbors=self.n_neighbors, extra=extra)
        return self

    def predict_one(self, x: dict):
        """Predict the label of a set of features `x`.
        Parameters:
            x: A dictionary of features.
        Returns:
            The neighbors
        """
        return self.predict_proba_one(x)

    def predict_proba_one(self, x):
        """
        This is modified to just return the nearest, not try to calculate
        a prediction because we just want the points back.
        """
        return self.find_nearest(x=x, n_neighbors=self.n_neighbors)

    def reset(self) -> "NearestNeighbors":
        """
        Reset estimator
        """
        self._reset()
        return self

    def _reset(self) -> "NearestNeighbors":
        """
        Reset estimator maintained on the parent for subclasses
        """
        self.window = collections.deque(maxlen=self.window_size)
        return self

    def append(self, x, y=None, extra: [Tuple, list] = None):
        """
        Add a point to the window, optionally with extra metadata (at end).
        """
        self.window.append((x, y, *(extra or [])))

    def update(self, x, y, n_neighbors, extra: [Tuple, list] = None):
        """
        Update the window with a new point, only added if > min distance. A
        boolean (true/false) is returned to indicate if the point was added.
        An extra tuple of list of features is allowed to add to x. The input `x`
        to this function should just be the features (typically dict).
        """
        # Don't add VERY similar points to window
        nearest = self.find_nearest(x, n_neighbors)

        # Default to add is False
        added = False

        # If we have any points too similar, don't keep.
        # Distance always the last index, index can be 2 or 3 depending on save_uid
        distances = [n[-1] for n in nearest if n[-1] < self.min_distance_keep]
        if not distances:
            self.append(x, y=y, extra=extra)
            added = True
        return added

    def find_nearest(self, x, n_neighbors):
        """
        Returns the `n_neighbors` closest points to `x`, along with their distances.
        This function assumes the x is a tuple or list with x[0] having relevant
        data for the distance calculation.
        """
        # Compute the distances to each point in the window
        points = (
            (*p, self.distance_func(a=x, b=p[0], **self.distance_func_kwargs))
            for p in self.window
        )

        # Return the k closest points (last index is distance)
        return sorted(points, key=operator.itemgetter(-1))[:n_neighbors]

    def set_distance_func(
        self, distance_func: Callable, distance_func_kwargs: dict = None
    ):
        """
        This is a courtesy function to show that the distance function can be
        set at runtime or after the model init.
        """
        self.distance_func = distance_func
        self.distance_func_kwargs = distance_func_kwargs or {}

    def set_default_distance_func(self, distance_func_kwargs: dict = None):
        """
        Set the default distance function.
        """
        self.distance_func = utils.math.minkowski_distance

        # Case 1: the user defined kwargs, but too many for Minkowski
        if distance_func_kwargs and len(distance_func_kwargs) > 1:
            raise ValueError(
                "Default Minkowski p-norm only accepts one parameter, p.\n"
                "Found {}".format(distance_func_kwargs)
            )

        # Case 2: the user defined kwargs, but missing p for Minkowski
        if distance_func_kwargs and "p" not in distance_func_kwargs:
            raise ValueError("Minkowski p-norm only accepts one parameter, p.\n")

        # Case 3: the user defined kwargs, but p is wrong
        if distance_func_kwargs and distance_func_kwargs["p"] < 1:
            raise ValueError(
                "Invalid Minkowski p-norm value: {}.\n"
                "Values must be greater than or equal to 1".format(
                    distance_func_kwargs["p"]
                )
            )

        # Case 4: No kwargs, set default
        if not distance_func_kwargs:
            distance_func_kwargs = {"p": 2.0}
        self.distance_func_kwargs = distance_func_kwargs
