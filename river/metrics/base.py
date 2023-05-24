from __future__ import annotations

import abc
import collections
import numbers
import operator

from river import base, stats, utils

from . import confusion

__all__ = [
    "BinaryMetric",
    "ClassificationMetric",
    "Metric",
    "Metrics",
    "MultiClassMetric",
    "RegressionMetric",
    "WrapperMetric",
]


class Metric(base.Base, abc.ABC):
    """Mother class for all metrics."""

    @abc.abstractmethod
    def update(self, y_true, y_pred) -> Metric:
        """Update the metric."""

    @abc.abstractmethod
    def revert(self, y_true, y_pred) -> Metric:
        """Revert the metric."""

    @abc.abstractmethod
    def get(self) -> float:
        """Return the current value of the metric."""

    @property
    @abc.abstractmethod
    def bigger_is_better(self) -> bool:
        """Indicate if a high value is better than a low one or not."""

    @abc.abstractmethod
    def works_with(self, model: base.Estimator) -> bool:
        """Indicates whether or not a metric can work with a given model."""

    @property
    def works_with_weights(self) -> bool:
        """Indicate whether the model takes into consideration the effect of sample weights"""
        return True

    def is_better_than(self, other) -> bool:
        """Indicate if the current metric is better than another one."""
        op = operator.gt if self.bigger_is_better else operator.lt
        return op(self.get(), other.get())

    def __gt__(self, other):
        return self.is_better_than(other)

    def __repr__(self):
        """Return the class name along with the current value of the metric."""
        return f"{self.__class__.__name__}: {self.get():{self._fmt}}".rstrip("0")

    def __str__(self):
        return repr(self)


class ClassificationMetric(Metric):
    """Mother class for all classification metrics.

    Parameters
    ----------
    cm
        This parameter allows sharing the same confusion
        matrix between multiple metrics. Sharing a confusion matrix reduces the amount of storage
        and computation time.

    """

    _fmt = ".2%"  # output a percentage, e.g. 0.427 becomes "42,7%"

    def __init__(self, cm: confusion.ConfusionMatrix | None = None):
        if cm is None:
            cm = confusion.ConfusionMatrix()
        self.cm = cm

    def update(self, y_true, y_pred, sample_weight=1.0):
        self.cm.update(
            y_true,
            y_pred,
            sample_weight=sample_weight,
        )
        return self

    def revert(self, y_true, y_pred, sample_weight=1.0):
        self.cm.revert(
            y_true,
            y_pred,
            sample_weight=sample_weight,
        )
        return self

    @property
    def bigger_is_better(self):
        return True

    def works_with(self, model) -> bool:
        return utils.inspect.isclassifier(model)

    @property
    def requires_labels(self):
        """Indicates if labels are required, rather than probabilities."""
        return True

    @staticmethod
    def _clamp_proba(p):
        """Clamp a number in between the (0, 1) interval."""
        return utils.math.clamp(p, minimum=1e-15, maximum=1 - 1e-15)

    def __add__(self, other) -> Metrics:
        if not isinstance(other, ClassificationMetric):
            raise ValueError(
                f"{self.__class__.__name__} and {other.__class__.__name__} metrics "
                "are not compatible"
            )
        return Metrics([self, other])


class BinaryMetric(ClassificationMetric):
    """Mother class for all binary classification metrics.

    Parameters
    ----------
    cm
        This parameter allows sharing the same confusion
        matrix between multiple metrics. Sharing a confusion matrix reduces the amount of storage
        and computation time.
    pos_val
        Value to treat as "positive".

    """

    def __init__(self, cm=None, pos_val=True):
        super().__init__(cm)
        self.pos_val = pos_val

    def update(
        self,
        y_true: bool,
        y_pred: bool | float | dict[bool, float],
        sample_weight=1.0,
    ) -> BinaryMetric:
        if self.requires_labels:
            y_pred = y_pred == self.pos_val
        return super().update(y_true == self.pos_val, y_pred, sample_weight)

    def revert(
        self,
        y_true: bool,
        y_pred: bool | float | dict[bool, float],
        sample_weight=1.0,
    ) -> BinaryMetric:
        if self.requires_labels:
            y_pred = y_pred == self.pos_val
        return super().revert(y_true == self.pos_val, y_pred, sample_weight)


class MultiClassMetric(ClassificationMetric):
    """Mother class for all multi-class classification metrics.

    Parameters
    ----------
    cm
        This parameter allows sharing the same confusion
        matrix between multiple metrics. Sharing a confusion matrix reduces the amount of storage
        and computation time.

    """

    def works_with(self, model) -> bool:
        return utils.inspect.isclassifier(model) or utils.inspect.isclusterer(model)


class RegressionMetric(Metric):
    """Mother class for all regression metrics."""

    _fmt = ",.6f"  # use commas to separate big numbers and show 6 decimals

    @abc.abstractmethod
    def update(self, y_true: numbers.Number, y_pred: numbers.Number) -> RegressionMetric:
        """Update the metric."""

    @abc.abstractmethod
    def revert(self, y_true: numbers.Number, y_pred: numbers.Number) -> RegressionMetric:
        """Revert the metric."""

    @property
    def bigger_is_better(self):
        return False

    def works_with(self, model) -> bool:
        return utils.inspect.isregressor(model)

    def __add__(self, other) -> Metrics:
        if not isinstance(other, RegressionMetric):
            raise ValueError(
                f"{self.__class__.__name__} and {other.__class__.__name__} metrics "
                "are not compatible"
            )
        return Metrics([self, other])


class Metrics(Metric, collections.UserList):
    """A container class for handling multiple metrics at once.

    Parameters
    ----------
    metrics
    str_sep

    """

    def __init__(self, metrics, str_sep=", "):
        super().__init__(metrics)
        self.str_sep = str_sep

    def update(self, y_true, y_pred, sample_weight=1.0):
        # If the metrics are classification metrics, then we have to handle the case where some
        # of the metrics require labels, whilst others need to be fed probabilities
        if hasattr(self, "requires_labels") and not self.requires_labels:
            for m in self:
                if m.requires_labels:
                    m.update(y_true, max(y_pred, key=y_pred.get))
                else:
                    m.update(y_true, y_pred)
            return self

        for m in self:
            m.update(y_true, y_pred)
        return self

    def revert(self, y_true, y_pred, sample_weight=1.0):
        # If the metrics are classification metrics, then we have to handle the case where some
        # of the metrics require labels, whilst others need to be fed probabilities
        if hasattr(self, "requires_labels") and not self.requires_labels:
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
        return all(m.works_with(model) for m in self)

    @property
    def bigger_is_better(self):
        raise NotImplementedError

    @property
    def works_with_weights(self):
        return all(m.works_with_weights for m in self)

    @property
    def requires_labels(self):
        return all(m.requires_labels for m in self)

    def __repr__(self):
        return self.str_sep.join(str(m) for m in self)

    def __add__(self, other):
        try:
            other + self[0]  # Will raise a ValueError if incompatible
        except IndexError:
            pass
        self.append(other)
        return self

    def clone(self):
        return self.__class__([m.clone() for m in self])


class WrapperMetric(Metric):
    @property
    def _fmt(self):
        return self.metric._fmt

    @property
    @abc.abstractmethod
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
        name = self.__class__.__name__
        return super().__repr__().replace(name, f"{name}({self.metric.__class__.__name__})")


class MeanMetric(abc.ABC):
    """Many metrics are just running averages. This is a utility class that avoids repeating
    tedious stuff throughout the module for such metrics.

    """

    def __init__(self):
        self._mean = stats.Mean()

    @abc.abstractmethod
    def _eval(self, y_true, y_pred):
        pass

    def update(self, y_true, y_pred, sample_weight=1.0):
        self._mean.update(x=self._eval(y_true, y_pred), w=sample_weight)
        return self

    def revert(self, y_true, y_pred, sample_weight=1.0):
        self._mean.revert(x=self._eval(y_true, y_pred), w=sample_weight)
        return self

    def get(self):
        return self._mean.get()


class ClusteringMetric(base.Base, abc.ABC):
    """
    Mother class of all internal clustering metrics.
    """

    # Define the format specification used for string representation.
    _fmt = ",.6f"  # Use commas to separate big numbers and show 6 decimals

    @abc.abstractmethod
    def update(self, x, y_pred, centers, sample_weight=1.0) -> ClusteringMetric:
        """Update the metric."""

    @abc.abstractmethod
    def revert(self, x, y_pred, centers, sample_weight=1.0) -> ClusteringMetric:
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
