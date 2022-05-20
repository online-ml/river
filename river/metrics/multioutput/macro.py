import statistics
from collections import defaultdict
from copy import deepcopy
from functools import partial

from river import metrics, utils
from river.metrics.multioutput.base import MultiOutputMetric

__all__ = ["MacroAverage"]


class MacroAverage(MultiOutputMetric, metrics.base.WrapperMetric):
    """Macro-average wrapper.

    A copy of the provided metric is made for each output. The arithmetic average of all the
    metrics is returned.

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
        for i in y_pred:
            self.metrics[i].update(y_true[i], y_pred[i], sample_weight)
        return self

    def revert(self, y_true, y_pred, sample_weight=1.0):
        for i in y_pred:
            self.metrics[i].revert(y_true[i], y_pred[i], sample_weight)
        return self

    def get(self):
        return statistics.mean(metric.get() for metric in self.metrics.values())
