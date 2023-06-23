from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from functools import partial

from river import metrics, utils
from river.metrics.multioutput.base import MultiOutputMetric

__all__ = ["PerOutput"]


class PerOutput(MultiOutputMetric, metrics.base.WrapperMetric):
    """Per-output wrapper.

    A copy of the metric is maintained for each output.

    Parameters
    ----------
    metric
        A classification or a regression metric.

    """

    def __init__(self, metric):
        self._metric = metric
        self.metrics = defaultdict(partial(deepcopy, self._metric))

    @property
    def metric(self):
        return self._metric

    def works_with(self, model) -> bool:
        if isinstance(self.metric, metrics.base.ClassificationMetric):
            return utils.inspect.ismoclassifier(model)
        return utils.inspect.ismoregressor(model)

    def update(self, y_true, y_pred, sample_weight=1.0):
        for i in y_true:
            self.metrics[i].update(y_true[i], y_pred[i], sample_weight)
        return self

    def revert(self, y_true, y_pred, sample_weight=1.0):
        for i in y_true:
            self.metrics[i].revert(y_true[i], y_pred[i], sample_weight)
        return self

    def get(self):
        return dict(self.metrics)

    def __repr__(self):
        return "\n".join(f"{i} - {metric}" for i, metric in self.metrics.items())
