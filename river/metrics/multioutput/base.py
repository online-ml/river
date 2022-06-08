import abc
import typing

from river import base, utils
from river.metrics.base import Metric

from .confusion import MultiLabelConfusionMatrix

__all__ = ["MultiOutputClassificationMetric", "MultiOutputRegressionMetric"]


class MultiOutputMetric(Metric):
    """Mother class for all multi-output metrics."""


class MultiOutputClassificationMetric(MultiOutputMetric):
    """Mother class for all multi-output classification metrics.

    Parameters
    ----------
    cm
        This parameter allows sharing the same confusion matrix between multiple metrics. Sharing a
        confusion matrix reduces the amount of storage and computation time.

    """

    _fmt = ".2%"

    def __init__(self, cm: MultiLabelConfusionMatrix = None):
        if cm is None:
            cm = MultiLabelConfusionMatrix()
        self.cm = cm

    def update(
        self,
        y_true: typing.Dict[typing.Union[str, int], base.typing.ClfTarget],
        y_pred: typing.Union[
            typing.Dict[typing.Union[str, int], base.typing.ClfTarget],
            typing.Dict[
                typing.Union[str, int],
                typing.Dict[base.typing.ClfTarget, float],
            ],
        ],
        sample_weight=1.0,
    ) -> "MultiOutputClassificationMetric":
        """Update the metric."""
        self.cm.update(y_true, y_pred, sample_weight)
        return self

    def revert(
        self,
        y_true: typing.Dict[typing.Union[str, int], base.typing.ClfTarget],
        y_pred: typing.Union[
            typing.Dict[typing.Union[str, int], base.typing.ClfTarget],
            typing.Dict[
                typing.Union[str, int],
                typing.Dict[base.typing.ClfTarget, float],
            ],
        ],
        sample_weight=1.0,
    ) -> "MultiOutputClassificationMetric":
        """Revert the metric."""
        self.cm.revert(y_true, y_pred, sample_weight)
        return self

    def works_with(self, model) -> bool:
        return utils.inspect.ismoclassifier(model)

    @property
    def bigger_is_better(self):
        return True

    @property
    def requires_labels(self):
        return True


class MultiOutputRegressionMetric(Metric):
    """Mother class for all multi-output regression metrics."""

    @abc.abstractmethod
    def update(
        self,
        y_true: typing.Dict[typing.Union[str, int], typing.Union[float, int]],
        y_pred: typing.Dict[typing.Union[str, int], typing.Union[float, int]],
    ) -> "MultiOutputRegressionMetric":
        """Update the metric."""

    @abc.abstractmethod
    def revert(
        self,
        y_true: typing.Dict[typing.Union[str, int], typing.Union[float, int]],
        y_pred: typing.Dict[typing.Union[str, int], typing.Union[float, int]],
    ) -> "MultiOutputRegressionMetric":
        """Revert the metric."""

    def works_with(self, model) -> bool:
        return utils.inspect.ismoregressor(model)

    @property
    def bigger_is_better(self):
        return False
