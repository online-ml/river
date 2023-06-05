from __future__ import annotations

import abc
import typing

from river import base


class DistanceFunc(typing.Protocol):
    def __call__(self, a: typing.Any, b: typing.Any, **kwargs) -> float:
        ...


class BaseNN(base.Estimator, abc.ABC):
    dist_func = None

    @abc.abstractmethod
    def append(self, item: typing.Any, **kwargs):
        pass

    @abc.abstractmethod
    def search(self, item: typing.Any, n_neighbors: int, **kwargs) -> tuple[list, list]:
        pass
