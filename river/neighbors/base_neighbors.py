import collections
import functools
import operator
from typing import Any, Callable, Tuple

from river import base, utils

__all__ = ["NearestNeighbors", "BaseKNN"]

DistanceFunc = Callable[[Any, Any], float]


def default_distance_func(a, b, **kwargs):
    """
    The default distance function, which is a wrapper to Minkowski distance.
    This serves as an example for those that want to implement their own functions.
    Since the "NearestNeighbors" class is generic to accept an "item" (typically
    a tuple (x,y) you cannot just provide the x, y as inputs, you need to
    provide the index into your item that includes the custom data (e.g., x[0]).
    The implementer (@vsoch) did not like this implementation because a distance
    function should not need to do extra parsing of the inputs X1 and X2,
    but wanted to compromise and be flexible since others liked the idea.
    """
    return utils.math.minkowski_distance(a[0], b[0], **kwargs)


class NearestNeighbors:
    """
    A basic data structure to hold nearest neighbors.

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
        distance_func_kwargs: dict = None,
    ):
        self.n_neighbors = n_neighbors
        self.window_size = window_size

        # A minimum distance (similarity) to determine adding to window
        # The model will perform better with a more diverse window
        # Since the distance function can be anything, it could be < 0
        self.min_distance_keep = min_distance_keep

        # If we don't have a distance func, set to use minkowski distance
        if not distance_func:
            self._set_default_distance_func(distance_func_kwargs)
        else:
            self._set_distance_func(distance_func, distance_func_kwargs)
        self.reset()

    def append(self, item: Any, extra: [Tuple, list] = None):
        """
        Add a point to the window, optionally with extra metadata.

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
        """
        Update the window with a new point, only added if > min distance.
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
        """
        Returns the `n_neighbors` closest points to `x`, along with their distances.
        This function assumes the x is a tuple or list with x[0] having relevant
        data for the distance calculation.
        """
        # Compute the distances to each point in the window
        # Item is JUST the (x,y) however the window is (item, <extra>, distance)
        points = ((*p, self.distance_func(item, p[0])) for p in self.window)

        # Return the k closest points (last index is distance)
        return sorted(points, key=operator.itemgetter(-1))[:n_neighbors]

    def reset(self) -> "NearestNeighbors":
        """
        Reset window
        """
        self.window = collections.deque(maxlen=self.window_size)

    def _set_distance_func(
        self, distance_func: Callable, distance_func_kwargs: dict = None
    ):
        """
        This is a courtesy function to show that the distance function can be
        set at runtime or after the model init.
        """
        self.distance_func = functools.partial(distance_func, **distance_func_kwargs)
        self.distance_func_kwargs = distance_func_kwargs

    def _set_default_distance_func(self, distance_func_kwargs: dict = None):
        """
        Set the default distance function. Since Minkowski distance is explicit
        about the range of p (and other parameters allowed) we do sanity checks
        here to ensure the kwargs provided only include p and p > 1
        """
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
        self._set_distance_func(default_distance_func, distance_func_kwargs)


class BaseKNN(base.Classifier):
    """
    K-Nearest Neighbors (KNN) to get neighbors back. This is the model you should
    use to save and retrieve custom metadata. It is also a base class for the
    other family of neighbors. It extends the base.Classifier but allows for
    multiclass prediction.

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
        An optional distance function that should be a callable with two inputs
        of Any type (e.g., a comparison A against B) that returns a distance.
        If not defined, the default Minkowski distance is used.

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

    >>> model = neighbors.BaseKNN(window_size=50)
    >>> for x, y in dataset.take(10):
    ...     model = model.learn_one(x, y)

    >>> # Add an extra vector of metadata to get back (e.g. a UID)
    >>> # If you have named features, use a dict in the list or tuple
    >>> model = neighbors.BaseKNN(window_size=50)
    >>> for x, y in dataset.take(10):
    ...     model = model.learn_one(x, y, extra=[str(uuid.uuid4())])

    When using this class, the "x" array for the window
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
        # These are not needed here, but the CI will fail without them
        self.n_neighbors = n_neighbors
        self.window_size = window_size
        self.min_distance_keep = min_distance_keep
        self.distance_func = distance_func
        self.distance_func_kwargs = distance_func_kwargs

        self.nn = NearestNeighbors(
            window_size=window_size,
            distance_func=distance_func,
            distance_func_kwargs=distance_func_kwargs,
            min_distance_keep=min_distance_keep,
            n_neighbors=n_neighbors,
        )

    @property
    def _multiclass(self):
        return True

    def learn_one(self, x, y=None, extra: [Tuple, list] = None):
        """Learn a set of features `x` and optional class `y`.
        Parameters:
            x: A dictionary of features.
            y: A class (optional if known).
            extra: an optional list or tuple of features to store
        Returns:
            self
        """
        self.nn.update((x, y), n_neighbors=self.n_neighbors, extra=extra)
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
        return self.nn.find_nearest((x, None), n_neighbors=self.n_neighbors)
