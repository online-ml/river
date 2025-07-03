from __future__ import annotations

import abc

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

    def __init__(self, cm: MultiLabelConfusionMatrix | None = None):
        if cm is None:
            cm = MultiLabelConfusionMatrix()
        self.cm = cm

    def update(
        self,
        y_true: dict[str | int, base.typing.ClfTarget],
        y_pred: dict[str | int, base.typing.ClfTarget]
        | dict[str | int, dict[base.typing.ClfTarget, float]],
        w=1.0,
    ) -> None:
        """Update the metric."""
        self.cm.update(y_true, y_pred, w)

    def revert(
        self,
        y_true: dict[str | int, base.typing.ClfTarget],
        y_pred: dict[str | int, base.typing.ClfTarget]
        | dict[str | int, dict[base.typing.ClfTarget, float]],
        w=1.0,
    ) -> None:
        """Revert the metric."""
        self.cm.revert(y_true, y_pred, w)

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
        y_true: dict[str | int, float | int],
        y_pred: dict[str | int, float | int],
    ) -> None:
        """Update the metric."""

    @abc.abstractmethod
    def revert(
        self,
        y_true: dict[str | int, float | int],
        y_pred: dict[str | int, float | int],
    ) -> None:
        """Revert the metric."""

    def works_with(self, model) -> bool:
        return utils.inspect.ismoregressor(model)

    @property
    def bigger_is_better(self):
        return False
