import abc
import numbers
import typing

from river import base, stats, utils

__all__ = ["InternalMetric"]


class InternalMetric(abc.ABC):
    """
    Mother class of all internal clustering metrics.
    """

    # Define the format specification used for string representation.
    _fmt = ",.6f"  # Use commas to separate big numbers and show 6 decimals

    @abc.abstractmethod
    def update(self, method, point, label, sample_weight) -> "InternalMetric":
        """Update the metric."""

    @abc.abstractmethod
    def revert(self, method, point, label, sample_weight) -> "InternalMetric":
        """Revert the metric."""

    @abc.abstractmethod
    def get(self) -> float:
        """Return the current value of the metric."""

    @abc.abstractmethod
    def bigger_is_better(self) -> bool:
        """Indicates if a high value is better than a low one or not."""

    @abc.abstractmethod
    def works_with(self, model: base.Estimator) -> bool:
        """Indicates whether or not a metric can work with a given model."""

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
        method,
        point: typing.Dict[typing.Union[str, int], typing.Union[float, int]],
        label: numbers.Number,
    ):
        pass

    def update(
        self,
        method,
        point: typing.Dict[typing.Union[str, int], typing.Union[float, int]],
        label: numbers.Number,
        sample_weight=1.0,
    ):
        self._mean.update(x=self._eval(method, point, label), w=sample_weight)
        return self

    def revert(
        self,
        method,
        point: typing.Dict[typing.Union[str, int], typing.Union[float, int]],
        label: numbers.Number,
        sample_weight=1.0,
    ):
        self._mean.revert(x=self._eval(method, point, label), w=sample_weight)
        return self

    def get(self):
        return self._mean.get()

    @property
    def bigger_is_better(self):
        return False

    def works_with(self, model) -> bool:
        return utils.inspect.isclusterer(model)
