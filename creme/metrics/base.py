import abc
import collections
import typing

from .. import base
from .. import utils
from .. import stats
from ..reco.base import Recommender


__all__ = [
    'BinaryMetric',
    'ClassificationMetric',
    'Metric',
    'Metrics',
    'MultiClassMetric',
    'MultiOutputClassificationMetric',
    'MultiOutputRegressionMetric',
    'RegressionMetric',
    'WrapperMetric'
]


class Metric(abc.ABC):

    # Define the format specification used for string representation.
    fmt = ',.6f'  # Use commas to separate big numbers and show 6 decimals

    @abc.abstractmethod
    def get(self) -> float:
        """Returns the current value of the metric."""

    @abc.abstractproperty
    def bigger_is_better(self) -> bool:
        """Indicates if a high value is better than a low one or not."""

    @abc.abstractmethod
    def works_with(self, model: base.Estimator) -> bool:
        """Indicates whether or not a metric can work with a given model."""

    def __repr__(self):
        """Returns the class name along with the current value of the metric."""
        return f'{self.__class__.__name__}: {self.get():{self.fmt}}'.rstrip('0')


class ClassificationMetric(Metric):

    @abc.abstractproperty
    def requires_labels(self) -> bool:
        """Helps to indicate if labels are required instead of probabilities."""

    @staticmethod
    def _clamp_proba(p):
        return utils.math.clamp(p, minimum=1e-15, maximum=1 - 1e-15)

    def __add__(self, other) -> 'Metrics':
        if not isinstance(other, ClassificationMetric):
            raise ValueError(f'{self.__class__.__name__} and {other.__class__.__name__} metrics '
                             'are not compatible')
        return Metrics([self, other])


class BinaryMetric(ClassificationMetric):

    @abc.abstractmethod
    def update(
        self,
        y_true: bool,
        y_pred: typing.Union[bool, float, typing.Dict[bool, float]],
        sample_weight: typing.Union[float, int]
    ) -> 'BinaryMetric':
        """Update the metric."""

    @abc.abstractmethod
    def revert(
        self,
        y_true: bool,
        y_pred: typing.Union[bool, float, typing.Dict[bool, float]],
        sample_weight: typing.Union[float, int]
    ) -> 'BinaryMetric':
        """Revert the metric."""

    def works_with(self, model) -> bool:
        return isinstance(utils.estimator_checks.guess_model(model), base.Classifier)


class MultiClassMetric(ClassificationMetric):

    @abc.abstractmethod
    def update(
        self,
        y_true: typing.Hashable,
        y_pred: typing.Union[typing.Hashable, typing.Dict[typing.Hashable, float]],
        sample_weight: typing.Union[float, int]
    ) -> 'MultiClassMetric':
        """Update the metric."""

    @abc.abstractmethod
    def revert(
        self,
        y_true: bool,
        y_pred: typing.Union[typing.Hashable, typing.Dict[typing.Hashable, float]],
        sample_weight: typing.Union[float, int]
    ) -> 'MultiClassMetric':
        """Revert the metric."""

    def works_with(self, model) -> bool:
        return isinstance(utils.estimator_checks.guess_model(model), base.Classifier)


class RegressionMetric(Metric):

    @abc.abstractmethod
    def update(
        self,
        y_true: typing.Union[float, int],
        y_pred: typing.Union[float, int],
        sample_weight: typing.Union[float, int]
    ) -> 'RegressionMetric':
        """Update the metric."""

    @abc.abstractmethod
    def revert(
        self,
        y_true: typing.Union[float, int],
        y_pred: typing.Union[float, int],
        sample_weight: typing.Union[float, int]
    ) -> 'RegressionMetric':
        """Revert the metric."""

    @property
    def bigger_is_better(self):
        return False

    def works_with(self, model) -> bool:
        return isinstance(utils.estimator_checks.guess_model(model), base.Regressor)

    def __add__(self, other) -> 'Metrics':
        if not isinstance(other, RegressionMetric):
            raise ValueError(f'{self.__class__.__name__} and {other.__class__.__name__} metrics '
                             'are not compatible')
        return Metrics([self, other])


class MultiOutputClassificationMetric(Metric):

    def update(
        self,
        y_true: typing.Dict[typing.Hashable, typing.Hashable],
        y_pred: typing.Union[
            typing.Dict[typing.Hashable, typing.Hashable],
            typing.Dict[typing.Hashable, typing.Dict[typing.Hashable, float]]
        ],
        sample_weight: typing.Union[float, int]
    ) -> 'MultiOutputClassificationMetric':
        """Update the metric."""

    def revert(
        self,
        y_true: typing.Dict[typing.Hashable, typing.Hashable],
        y_pred: typing.Union[
            typing.Dict[typing.Hashable, typing.Hashable],
            typing.Dict[typing.Hashable, typing.Dict[typing.Hashable, float]]
        ],
        sample_weight: typing.Union[float, int]
    ) -> 'MultiOutputClassificationMetric':
        """Revert the metric."""

    def works_with(self, model) -> bool:
        return isinstance(utils.estimator_checks.guess_model(model), base.MultiOutputClassifier)


class MultiOutputRegressionMetric(Metric):

    def update(
        self,
        y_true: typing.Dict[typing.Hashable, typing.Union[float, int]],
        y_pred: typing.Dict[typing.Hashable, typing.Union[float, int]],
        sample_weight: typing.Union[float, int]
    ) -> 'MultiOutputRegressionMetric':
        """Update the metric."""

    def revert(
        self,
        y_true: typing.Dict[typing.Hashable, typing.Union[float, int]],
        y_pred: typing.Dict[typing.Hashable, typing.Union[float, int]],
        sample_weight: typing.Union[float, int]
    ) -> 'MultiOutputRegressionMetric':
        """Revert the metric."""

    def works_with(self, model) -> bool:
        return isinstance(utils.estimator_checks.guess_model(model), base.MultiOutputRegressor)


class Metrics(Metric, collections.UserList):
    """A container class for handling multiple metrics at once."""

    def __init__(self, metrics, str_sep=', '):
        super().__init__(metrics)
        self.str_sep = str_sep

    def update(self, y_true, y_pred, sample_weight=1.):

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

    def revert(self, y_true, y_pred, sample_weight=1.):

        # If the metrics are classification metrics, then we have to handle the case where some
        # of the metrics require labels, whilst others need to be fed probabilities
        if hasattr(self, 'requires_labels') and not self.requires_labels:
            for m in self:
                if m.requires_labels:
                    m.revert(y_true, max(y_pred, key=y_pred.get), sample_weight)
                else:
                    m.revert(y_true, y_pred, sample_weight)
            return self

        for m in self:
            m.revert(y_true, y_pred, sample_weight)
        return self

    def get(self):
        return [m.get() for m in self]

    def works_with(self, model) -> bool:
        return all(m.works_with(utils.estimator_checks.guess_model(model)) for m in self)

    @property
    def bigger_is_better(self):
        raise NotImplementedError

    @property
    def requires_labels(self):
        return all(m.requires_labels for m in self)

    def __repr__(self):
        return self.str_sep.join((str(m) for m in self))

    def __add__(self, other):
        try:
            other + self[0]  # Will raise a ValueError if incompatible
        except IndexError:
            pass
        self.append(other)
        return self


class WrapperMetric(Metric):

    @property
    def fmt(self):
        return self.metric.fmt

    @abc.abstractproperty
    def metric(self):
        """Gives access to the wrapped metric."""

    def get(self):
        return self.metric.get()

    @property
    def bigger_is_better(self):
        return self.metric.bigger_is_better

    def works_with(self, model):
        return self.metric.works_with(model)

    @property
    def requires_labels(self):
        return self.metric.requires_labels

    @property
    def __metaclass__(self):
        return self.metric.__class__

    def __repr__(self):
        return str(self.metric)


class MeanMetric(abc.ABC):
    """Many metrics are just running averages. This is a utility class that avoids repeating
    tedious stuff throughout the module for such metrics.

    """

    def __init__(self):
        self._mean = stats.Mean()

    @abc.abstractmethod
    def _eval(self, y_true, y_pred):
        pass

    def update(self, y_true, y_pred, sample_weight=1.):
        self._mean.update(x=self._eval(y_true, y_pred), w=sample_weight)
        return self

    def revert(self, y_true, y_pred, sample_weight=1.):
        self._mean.revert(x=self._eval(y_true, y_pred), w=sample_weight)
        return self

    def get(self):
        return self._mean.get()
