import abc
import collections
import typing

from .. import base
from .. import utils


__all__ = [
    'BinaryMetric',
    'MultiClassMetric',
    'RegressionMetric'
]


class Metric(abc.ABC):

    @abc.abstractmethod
    def get(self) -> float:
        """Returns the current value of the metric."""

    @property
    @abc.abstractmethod
    def bigger_is_better(self) -> bool:
        """Indicates if a high value is better than a low one or not."""

    @abc.abstractmethod
    def works_with(self, model) -> bool:
        """Indicates whether or not a metric can work with a given model."""

    def __str__(self):
        """Returns the class name along with the current value of the metric."""
        return f'{self.__class__.__name__}: {self.get():.6f}'.rstrip('0')

    def __repr__(self):
        return str(self)


class ClassificationMetric(Metric):

    @property
    @abc.abstractmethod
    def requires_labels(self) -> bool:
        """Helps to indicate if labels are required instead of probabilities."""

    @staticmethod
    def clamp_proba(p):
        return utils.clamp(p, minimum=1e-15, maximum=1 - 1e-15)

    def __add__(self, other) -> 'Metrics':
        if not isinstance(other, ClassificationMetric):
            raise ValueError(f'{self.__class__.__name__} and {other.__class__.__name__} metrics '
                             'are not compatible')
        return Metrics([self, other])


class BinaryMetric(ClassificationMetric):

    @abc.abstractmethod
    def update(self, y_true: bool, y_pred: typing.Union[bool, base.Probas]) -> 'BinaryMetric':
        """Updates the metric."""

    def works_with(self, model) -> bool:
        return isinstance(model, base.BinaryClassifier)


class MultiClassMetric(BinaryMetric):

    @abc.abstractmethod
    def update(self, y_true: base.Label,
               y_pred: typing.Union[base.Label, base.Probas]) -> 'MultiClassMetric':
        """Updates the metric."""

    def works_with(self, model) -> bool:
        return isinstance(model, base.Classifier)


class RegressionMetric(Metric):

    @abc.abstractmethod
    def update(self, y_true: float, y_pred: float) -> 'RegressionMetric':
        """Updates the metric."""

    @property
    def bigger_is_better(self):
        return False

    def works_with(self, model) -> bool:
        return isinstance(model, base.Regressor)

    def __add__(self, other) -> 'Metrics':
        if not isinstance(other, RegressionMetric):
            raise ValueError(f'{self.__class__.__name__} and {other.__class__.__name__} metrics '
                             'are not compatible')
        return Metrics([self, other])


class MultiOutputClassificationMetric(ClassificationMetric):

    def update(self, y_true: typing.Dict[str, base.Label],
               y_pred: typing.Dict[str, typing.Union[base.Label, base.Probas]]):
        """Updates the metric."""

    def works_with(self, model) -> bool:
        return isinstance(model, base.MultiOutputClassifier)


class MultiOutputRegressionMetric(RegressionMetric):

    def update(self, y_true: typing.Dict[str, float], y_pred: typing.Dict[str, float]):
        """Updates the metric."""

    def works_with(self, model) -> bool:
        return isinstance(model, base.MultiOutputRegressor)


class Metrics(Metric, collections.UserList):
    """A container class for handling multiple metrics at once."""

    def __init__(self, metrics, str_sep=', '):
        super().__init__(metrics)
        self.str_sep = str_sep

    def update(self, y_true, y_pred):

        # If the metrics are classification metrics, then we have to handle the case where some
        # of the metrics require labels, whilst others need to be fed probabilities
        if hasattr(self, 'requires_labels') and not self.requires_labels:
            for m in self:
                if m.requires_labels:
                    m.update(y_true, max(y_pred, key=y_pred.get))
                else:
                    m.update(y_true, y_pred)
            return self

        for m in self:
            m.update(y_true, y_pred)
        return self

    def get(self):
        return [m.get() for m in self]

    def works_with(self, model) -> bool:
        return all(m.works_with(model) for m in self)

    @property
    def bigger_is_better(self):
        raise NotImplementedError

    @property
    def requires_labels(self):
        return all(m.requires_labels for m in self)

    def __str__(self):
        return self.str_sep.join((str(m) for m in self))

    def __add__(self, other):
        try:
            other + self[0]  # Will raise a ValueError if incompatible
        except IndexError:
            pass
        self.append(other)
        return self
