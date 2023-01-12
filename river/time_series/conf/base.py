import abc
import numbers
import operator
import dataclasses

from river import base, stats, utils
from collections import deque

__all__=[
    "Interval",
    "Gaussian",
    "RegressionInterval"
]


# from river/metrics/base.py
@dataclasses.dataclass
class Interval(base.Base, abc.ABC):
    """Mother class for all intervals
    
    An object to represent a (prediction) interval.

    Users are not expected to use this class as-is. Instead, they should use the `with_interval`
    parameter of the `predict_one` method of any regressor or classifier wrapped with a conformal
    prediction method.

    Parameters
    ----------
    lower
        The lower bound of the interval.
    upper
        The upper bound of the interval.

    """

    lower: float
    upper: float

    # Begin : From initial interval.py in conf 
    @property
    def center(self):
        """The center of the interval."""
        return (self.lower + self.upper) / 2

    @property
    def width(self):
        """The width of the interval."""
        return self.upper - self.lower

    def __contains__(self, x):
        return self.lower <= x <= self.upper
    # End : From initial interval.py in conf 

    @abc.abstractmethod
    def update(self, y_true, y_pred) -> "Interval":
        """Update the Interval."""
        
    @abc.abstractmethod
    def get(self) -> tuple(float):
        """Return the current value of the Interval."""

    @property
    @abc.abstractmethod
    def bigger_is_better(self) -> bool:
        """Indicate if a high value is better than a low one or not."""

    @abc.abstractmethod
    def works_with(self, model: base.Estimator) -> bool:
        """Indicates whether or not a Interval can work with a given model."""

    @property
    def works_with_weights(self) -> bool:
        """Indicate whether the model takes into consideration the effect of sample weights"""
        return True

    def is_better_than(self, other) -> bool:
        op = operator.gt if self.bigger_is_better else operator.lt
        return op(self.get(), other.get())

    def __gt__(self, other):
        return self.is_better_than(other)

    def __repr__(self) -> str:
        """Return the class name along with the current value of the Interval."""
        return f"{self.__class__.__name__}: {self.get():{self._fmt}}".rstrip("0")

    def __str__(self) -> str:
        return repr(self)

    def __eq__(self, other) -> bool:
        """Compare two instance of interval"""
        if isinstance(other, Interval):
            return self.lower == other.lower and self.upper == other.upper
        return False
