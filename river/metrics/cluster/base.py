import abc
import numbers
import typing

from river import stats

__all__ = ["InternalClusMetric"]


class InternalClusMetric(abc.ABC):
    """
    Mother class of all internal clustering metrics.
    """

    # Define the format specification used for string representation.
    _fmt = ",.6f"  # Use commas to separate big numbers and show 6 decimals

    @abc.abstractmethod
    def update(
        self, x, y_pred, centers, sample_weight=1.0
    ) -> "InternalClusMetric":
        """Update the metric."""

    @abc.abstractmethod
    def revert(
        self, x, y_pred, centers, sample_weight=1.0
    ) -> "InternalClusMetric":
        """Revert the metric."""

    @abc.abstractmethod
    def get(self) -> float:
        """Return the current value of the metric."""

    @abc.abstractmethod
    def bigger_is_better(self) -> bool:
        """Indicates if a high value is better than a low one or not."""

    def __repr__(self):
        """Returns the class name along with the current value of the metric."""
        return f"{self.__class__.__name__}: {self.get():{self._fmt}}".rstrip("0")


class MeanInternalMetric(InternalClusMetric):
    """Many metrics are just running averages. This is a utility class that avoids repeating
    tedious stuff throughout the module for such metrics.

    """

    def __init__(self):
        self._mean = stats.Mean()

    @abc.abstractmethod
    def _eval(
        self,
        x: typing.Dict[typing.Union[str, int], typing.Union[float, int]],
        y_pred: numbers.Number,
        centers,
    ):
        pass

    def update(
        self,
        x: typing.Dict[typing.Union[str, int], typing.Union[float, int]],
        y_pred: numbers.Number,
        centers,
        sample_weight=1.0,
    ):
        self._mean.update(x=self._eval(x, y_pred, centers), w=sample_weight)
        return self

    def revert(
        self,
        x: typing.Dict[typing.Union[str, int], typing.Union[float, int]],
        y_pred: numbers.Number,
        centers,
        sample_weight=1.0,
    ):
        self._mean.revert(x=self._eval(x, y_pred, centers), w=sample_weight)
        return self

    def get(self):
        return self._mean.get()
