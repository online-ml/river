from typing import Tuple

from river import base

from .neighbors import DistanceFunc, MinkowskiNeighbors

__all__ = ["BaseKNN"]


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

    Here is an example with a custom item and distance function.
    >>> from dataclasses import dataclass
    >>> import functools
    >>> @dataclass
    ... class Entry:
    ...     name: str
    ...     rate: float
    ...     age: int

    ... def foo(self):
    ...     pass

    >>> def bar(self):
    ...     pass

    >>> def my_dist(a: Entry, b: Entry):
    ...     return (a[0].rate - b[0].rate) ** 2

    >>> def my_dist(a: Entry, b: Entry, coefficient: float = 1.0):
    ...     return coefficient * (a[0].rate - b[0].rate) ** 2

    >>> distance_func = functools.partial(my_dist, coefficient=42.0)
    >>> model = neighbors.BaseKNN(window_size=50, distance_func=distance_func)
    >>> # model.learn_one(Entry("marta", 5.0, 36))
    >>> # model.predict_one(Entry("harry", 10.0, 22))

    Note that when using a custom distance function, the values provided as a
    and b have the item you provided to learn at the first index (0). If you
    provide a value, y (class) that would be at index 1. After the
    metric is run, the array returned has the original item again at index
    0, followed by the items you provided in extra (default are None) and then
    the final value in the array is the distance.
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        window_size: int = 1000,
        min_distance_keep: float = 0.0,
        distance_func: DistanceFunc = None,
    ):
        # These are not needed here, but the CI will fail without them
        self.n_neighbors = n_neighbors
        self.window_size = window_size
        self.min_distance_keep = min_distance_keep
        self.distance_func = distance_func

        self.nn = MinkowskiNeighbors(
            window_size=window_size,
            distance_func=distance_func,
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
