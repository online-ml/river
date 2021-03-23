import abc
import numbers
import typing

from river import base, stats, utils
from river.base.typing import FeatureName

__all__ = ["InternalMetric"]


class InternalMetric(abc.ABC):
    """
    Mother class of all internal clustering metrics.
    """

    # Define the format specification used for string representation.
    _fmt = ",.6f"  # Use commas to separate big numbers and show 6 decimals

    @abc.abstractmethod
    def update(self, x, y_pred, centers, sample_weight=1.0) -> "InternalMetric":
        """Update the metric."""

    @abc.abstractmethod
    def revert(self, x, y_pred, centers, sample_weight=1.0) -> "InternalMetric":
        """Revert the metric."""

    @abc.abstractmethod
    def get(self) -> float:
        """Return the current value of the metric."""

    @property
    @abc.abstractmethod
    def bigger_is_better(self) -> bool:
        """Indicates if a high value is better than a low one or not."""

    def works_with(self, model: base.Estimator) -> bool:
        """Indicates whether or not a metric can work with a given model."""
        return utils.inspect.isclusterer(model)

    def __repr__(self):
        """Returns the class name along with the current value of the metric."""
        return f"{self.__class__.__name__}: {self.get():{self._fmt}}".rstrip("0")


class MeanInternalMetric(InternalMetric):
    """Many metrics are just running averages. This is a utility class that avoids repeating
    tedious stuff throughout the module for such metrics.

    """

    def __init__(self):
        self._mean = stats.Mean()

    @abc.abstractmethod
    def _eval(
        self,
        x: typing.Dict[FeatureName, numbers.Number],
        y_pred: numbers.Number,
        centers,
        sample_weight=1.0,
    ):
        pass

    def update(
        self,
        x: typing.Dict[FeatureName, numbers.Number],
        y_pred: numbers.Number,
        centers,
        sample_weight=1.0,
    ):
        self._mean.update(x=self._eval(x, y_pred, centers), w=sample_weight)
        return self

    def revert(
        self,
        x: typing.Dict[FeatureName, numbers.Number],
        y_pred: numbers.Number,
        centers,
        sample_weight=1.0,
    ):
        self._mean.revert(x=self._eval(x, y_pred, centers), w=sample_weight)
        return self

    def get(self):
        return self._mean.get()
