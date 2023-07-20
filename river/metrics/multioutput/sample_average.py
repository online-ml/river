from __future__ import annotations

from river import metrics, stats, utils
from river.metrics.multioutput.base import MultiOutputMetric

__all__ = ["SampleAverage"]


class SampleAverage(MultiOutputMetric, metrics.base.WrapperMetric):
    """Sample-average wrapper.

    The provided metric is evaluate on each sample. The arithmetic average over all the samples is
    returned. This is equivalent to using `average='samples'` in scikit-learn.

    Parameters
    ----------
    metric
        A classification or a regression metric.

    Examples
    --------

    >>> from river import metrics

    >>> y_true = [
    ...     {0: False, 1: True, 2: True},
    ...     {0: True, 1: True, 2: False}
    ... ]
    >>> y_pred = [
    ...     {0: True, 1: True, 2: True},
    ...     {0: True, 1: False, 2: False}
    ... ]

    >>> sample_jaccard = metrics.multioutput.SampleAverage(metrics.Jaccard())

    >>> for yt, yp in zip(y_true, y_pred):
    ...     sample_jaccard = sample_jaccard.update(yt, yp)
    >>> sample_jaccard
    SampleAverage(Jaccard): 58.33%

    """

    def __init__(self, metric):
        self._metric = metric
        self._avg = stats.Mean()

    @property
    def metric(self):
        return self._metric

    def works_with(self, model) -> bool:
        if isinstance(self.metric, metrics.base.ClassificationMetric):
            return utils.inspect.ismoclassifier(model)
        return utils.inspect.ismoregressor(model)

    def update(self, y_true, y_pred, sample_weight=1.0):
        metric = self.metric.clone()
        for i in y_true:
            metric.update(y_true[i], y_pred[i])
        self._avg.update(metric.get(), sample_weight)
        return self

    def revert(self, y_true, y_pred, sample_weight=1.0):
        metric = self.metric.clone()
        for i in y_true:
            metric.update(y_true[i], y_pred[i])
        self._avg.revert(metric.get(), sample_weight)
        return self

    def get(self):
        return self._avg.get()
