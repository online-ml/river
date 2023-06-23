from __future__ import annotations

import math

from river import metrics

__all__ = ["MSE", "RMSE", "RMSLE"]


class MSE(metrics.base.MeanMetric, metrics.base.RegressionMetric):
    """Mean squared error.

    Examples
    --------

    >>> from river import metrics

    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]

    >>> metric = metrics.MSE()

    >>> for yt, yp in zip(y_true, y_pred):
    ...     print(metric.update(yt, yp).get())
    0.25
    0.25
    0.1666
    0.375

    """

    def _eval(self, y_true, y_pred):
        return (y_true - y_pred) ** 2


class RMSE(MSE):
    """Root mean squared error.

    Examples
    --------

    >>> from river import metrics

    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]

    >>> metric = metrics.RMSE()
    >>> for yt, yp in zip(y_true, y_pred):
    ...     print(metric.update(yt, yp).get())
    0.5
    0.5
    0.408248
    0.612372

    >>> metric
    RMSE: 0.612372

    """

    def get(self):
        return super().get() ** 0.5


class RMSLE(RMSE):
    """Root mean squared logarithmic error.

    Examples
    --------

    >>> from river import metrics

    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]

    >>> metric = metrics.RMSLE()
    >>> for yt, yp in zip(y_true, y_pred):
    ...     metric = metric.update(yt, yp)

    >>> metric
    RMSLE: 0.357826

    """

    def update(self, y_true, y_pred, sample_weight=1.0):
        return super().update(math.log(y_true + 1), math.log(y_pred + 1), sample_weight)
