import abc
import collections
import numbers
import operator
import typing

from river import base, stats, utils

from . import confusion

# from river/metrics/base.py
class ConformalPrediction(base.Base, abc.ABC):
    """Mother class for all intervals."""

    @abc.abstractmethod
    def update(self, y_true, y_pred) -> "ConformalPrediction":
        """Update the conformal prediction."""

    @abc.abstractmethod
    def revert(self, y_true, y_pred) -> "ConformalPrediction":
        """Revert the conformal prediction."""

    @abc.abstractmethod
    def get(self) -> float:
        """Return the current value of the ConformalPrediction."""

    @property
    @abc.abstractmethod
    def bigger_is_better(self) -> bool:
        """Indicate if a high value is better than a low one or not."""

    @abc.abstractmethod
    def works_with(self, model: base.Estimator) -> bool:
        """Indicates whether or not a conformal prediction can work with a given model."""

    @property
    def works_with_weights(self) -> bool:
        """Indicate whether the model takes into consideration the effect of sample weights"""
        return True

    def is_better_than(self, other) -> bool:
        op = operator.gt if self.bigger_is_better else operator.lt
        return op(self.get(), other.get())

    def __gt__(self, other):
        return self.is_better_than(other)

    def __repr__(self):
        """Return the class name along with the current value of the conformal prediction."""
        return f"{self.__class__.__name__}: {self.get():{self._fmt}}".rstrip("0")

    def __str__(self):
        return repr(self)


class RegressionConformalPrediction(ConformalPrediction):
    """Mother class for all regression conformal prediction."""

    _fmt = ",.6f"  # use commas to separate big numbers and show 6 decimals

    @abc.abstractmethod
    def update(self, y_true: numbers.Number, y_pred: numbers.Number) -> "RegressionConformalPrediction":
        """Update the interval."""

    @abc.abstractmethod
    def revert(self, y_true: numbers.Number, y_pred: numbers.Number) -> "RegressionConformalPrediction":
        """Revert the interval."""

    @property
    def bigger_is_better(self):
        return False

    def works_with(self, model) -> bool:
        return utils.inspect.isregressor(model)

    def __add__(self, other) -> "ConformalPrediction":
        if not isinstance(other, RegressionConformalPrediction):
            raise ValueError(
                f"{self.__class__.__name__} and {other.__class__.__name__} ConformalPrediction "
                "are not compatible"
            )
        return ConformalPrediction([self, other])
