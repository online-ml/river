from __future__ import annotations

from river import metrics

__all__ = ["Accuracy"]


class Accuracy(metrics.base.MultiClassMetric):
    """Accuracy score, which is the percentage of exact matches.

    Parameters
    ----------
    cm
        This parameter allows sharing the same confusion
        matrix between multiple metrics. Sharing a confusion matrix reduces the amount of storage
        and computation time.

    Examples
    --------

    >>> from river import metrics

    >>> y_true = [True, False, True, True, True]
    >>> y_pred = [True, True, False, True, True]

    >>> metric = metrics.Accuracy()
    >>> for yt, yp in zip(y_true, y_pred):
    ...     metric = metric.update(yt, yp)

    >>> metric
    Accuracy: 60.00%

    """

    def get(self):
        try:
            return self.cm.total_true_positives / self.cm.total_weight
        except ZeroDivisionError:
            return 0.0
