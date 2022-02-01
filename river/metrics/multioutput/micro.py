from river import utils

from . import base
from ..base import WrapperMetric

__all__ = ["MicroAverage"]


class MicroAverage(base.MultiOutputMetric, WrapperMetric):
    """Micro-average wrapper.

    The provided metric is updated with the value of each output.

    Parameters
    ----------
    metric
        A classification or a regression metric.

    """

    def __init__(self, metric):
        self._metric = metric

    @property
    def metric(self):
        return self._metric

    def works_with(self, model) -> bool:
        if isinstance(self.metric, metrics.ClassificationMetric):
            return utils.inspect.ismoclassifier(model)
        return utils.inspect.ismoregressor(model)

    def update(self, y_true, y_pred, sample_weight=1.0):
        for i in y_true:
            self.metric.update(y_true[i], y_pred[i], sample_weight)
        return self

    def revert(self, y_true, y_pred, sample_weight=1.0):
        for i in y_true:
            self.metric.revert(y_true[i], y_pred[i], sample_weight)
        return self

    def get(self):
        return self.metric.get()
