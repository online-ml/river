from __future__ import annotations

import abc
import typing

from river import base
from river.utils.vectordict import (
    euclidean_distance_tuple as _euclidean_tuple_distance,  # noqa: F401
)

__all__ = ["BaseNN", "DistanceFunc", "FunctionWrapper", "_euclidean_tuple_distance"]


class DistanceFunc(typing.Protocol):
    def __call__(self, a: typing.Any, b: typing.Any, **kwargs: typing.Any) -> float: ...


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

    __slots__ = ("distance_function",)

    def __init__(self, distance_function: DistanceFunc) -> None:
        self.distance_function = distance_function

    def __call__(self, a: typing.Any, b: typing.Any) -> float:
        # Access x, which is stored in a tuple (x, y)
        return self.distance_function(a[0], b[0])


# TODO: Generic instead of Any for more precise typing
# should remove the type ignore comments KNNRegressor.predict_one
class BaseNN(base.Estimator, abc.ABC):
    def __init__(self, dist_func: DistanceFunc | FunctionWrapper) -> None:
        self.dist_func = dist_func

    @abc.abstractmethod
    def append(self, item: typing.Any, **kwargs: typing.Any) -> None:
        pass

    @abc.abstractmethod
    def search(
        self, item: typing.Any, n_neighbors: int, **kwargs: typing.Any
    ) -> tuple[list[typing.Any], list[float]]:
        pass

    @abc.abstractmethod
    def refresh_targets(self) -> set[base.typing.ClfTarget]:
        pass
