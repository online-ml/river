from __future__ import annotations

from river import metrics

__all__ = ["MAPE"]


class MAPE(metrics.base.MeanMetric, metrics.base.RegressionMetric):
    """Mean absolute percentage error.

    Examples
    --------

    >>> from river import metrics

    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]

    >>> metric = metrics.MAPE()
    >>> for yt, yp in zip(y_true, y_pred):
    ...     metric = metric.update(yt, yp)

    >>> metric
    MAPE: 32.738095

    """

    def _eval(self, y_true, y_pred):
        if y_true == 0:
            return 0.0
        return abs(y_true - y_pred) / abs(y_true)

    def get(self):
        return 100 * super().get()
