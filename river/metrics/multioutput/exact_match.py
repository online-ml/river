from river import metrics
from river.metrics.multioutput.base import MultiOutputClassificationMetric

__all__ = ["ExactMatch"]


class ExactMatch(metrics.base.MeanMetric, MultiOutputClassificationMetric):
    """Exact match score.

    This is the most strict multi-label metric, defined as the number of
    samples that have all their labels correctly classified, divided by the
    total number of samples.

    Parameters
    ----------
    cm
        This parameter allows sharing the same confusion matrix between multiple metrics. Sharing a
        confusion matrix reduces the amount of storage and computation time.

    Examples
    --------

    >>> from river import metrics

    >>> y_true = [
    ...     {0: False, 1: True, 2: True},
    ...     {0: True, 1: True, 2: False},
    ...     {0: True, 1: True, 2: False},
    ... ]

    >>> y_pred = [
    ...     {0: True, 1: True, 2: True},
    ...     {0: True, 1: False, 2: False},
    ...     {0: True, 1: True, 2: False},
    ... ]

    >>> metric = metrics.multioutput.ExactMatch()
    >>> for yt, yp in zip(y_true, y_pred):
    ...     metric = metric.update(yt, yp)

    >>> metric
    ExactMatch: 33.33%

    """

    def _eval(self, y_true, y_pred):
        return y_true == y_pred
