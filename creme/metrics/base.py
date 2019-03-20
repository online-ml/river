import abc

from .. import base
from .. import types


__all__ = [
    'BinaryClassificationMetric',
    'MultiClassificationMetric',
    'RegressionMetric'
]


class BaseMetric(abc.ABC):

    @abc.abstractmethod
    def get(self) -> float:
        """Returns the current value of the metric."""

    @abc.abstractmethod
    def works_with(self, model) -> bool:
        """Tells if a metric can work with a given model or not."""

    def __str__(self):
        """Returns the class name along with the current value of the metric."""
        return f'{self.__class__.__name__}: {self.get():.6f}'.rstrip('0')

    def __repr__(self):
        return str(self)


class ClassificationMetric(BaseMetric):

    @property
    @abc.abstractmethod
    def requires_labels(self):
        """Helps to indicate if labels are required instead of probabilities."""


class BinaryClassificationMetric(ClassificationMetric):

    @abc.abstractmethod
    def update(self, y_true: bool, y_pred: float):
        """Updates the metric."""

    def works_with(self, model):
        return isinstance(model, base.BinaryClassifier)


class MultiClassificationMetric(BinaryClassificationMetric):

    @abc.abstractmethod
    def update(self, y_true: types.Label, y_pred: dict):
        """Updates the metric."""

    def works_with(self, model):
        return isinstance(model, (base.MultiClassifier, base.BinaryClassifier))


class RegressionMetric(BaseMetric):

    @abc.abstractmethod
    def update(self, y_true: float, y_pred: float):
        """Updates the metric."""

    def works_with(self, model):
        return isinstance(model, base.Regressor)
