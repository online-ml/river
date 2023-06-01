import abc
import collections
import operator

from river import base

# from river.changepoints.methods.base import ChangePointDetector TODO: Change path for integration into river
from methods.base import ChangePointDetector


class ChangePointMetric:
    """Mother class for all change point detection metrics."""

    def __init__(self, margin=5):
        self.margin = margin
        self.value = None

    @abc.abstractmethod
    def __call__(self, annotations, predictions, **kwargs):
        """Execute the metric"""

    def get(self) -> float:
        """Return the current value of the metric."""
        return self.value

    @property
    def bigger_is_better(self):
        """Indicate if a high value is better than a low one or not."""
        return False

    @staticmethod
    def works_with(model: base.Estimator) -> bool:
        """Indicates whether or not a metric can work with a given model."""
        return isinstance(model, ChangePointDetector)

    @property
    def works_with_weights(self) -> bool:
        """Indicate whether the model takes into consideration the effect of sample weights"""
        return False

    def is_better_than(self, other) -> bool:
        op = operator.gt if self.bigger_is_better else operator.lt
        return op(self.get(), other.get())

    def __gt__(self, other):
        return self.is_better_than(other)

    def __repr__(self):
        """Return the class name"""
        return self.__class__.__name__

    def __str__(self):
        return repr(self)

    def __add__(self, other):
        return ChangePointMetrics([self, other])


class ChangePointMetrics(collections.UserList, ChangePointMetric):

    def __init__(self, metrics, str_sep=", "):
        super().__init__(metrics)
        self.str_sep = str_sep

    def __call__(self, annotations, predictions, **kwargs):
        values = []
        for metric in self:
            values.append(metric(annotations, predictions, **kwargs))
        self.value = values
        return values

    def bigger_is_better(self):
        return all(metric.bigger_is_better for metric in self)

    def is_better_than(self, other) -> bool:
        raise NotImplementedError("Cannot compare multiple metrics.")

    def __repr__(self):
        return self.str_sep.join(str(m) for m in self)
