from __future__ import annotations

import abc
import typing

from river import base


class DistanceFunc(typing.Protocol):
    def __call__(self, a: typing.Any, b: typing.Any, **kwargs) -> float:
        ...


class FunctionWrapper:
    """Wrapper used to make distance function work with KNNClassifier and
    KNNRegressor.

    The k-NN-based classifier and regressor store tuples with `(x, y)`, but only
    `x` is used for distance calculations. This wrapper makes sure `x` is accessed
    when calculating the distances.

    Parameters
    ----------
    distance_function
        The custom distance function to be wrapped.
    """

    def __init__(self, distance_function: DistanceFunc):
        self.distance_function = distance_function

    def __call__(self, a, b):
        # Access x, which is stored in a tuple (x, y)
        return self.distance_function(a[0], b[0])


class BaseNN(base.Estimator, abc.ABC):
    def __init__(self, dist_func: DistanceFunc | FunctionWrapper):
        self.dist_func = dist_func

    @abc.abstractmethod
    def append(self, item: typing.Any, **kwargs):
        pass

    @abc.abstractmethod
    def search(self, item: typing.Any, n_neighbors: int, **kwargs) -> tuple[list, list]:
        pass
