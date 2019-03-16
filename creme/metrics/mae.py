from .. import stats

from . import base


class MAE(stats.Mean, base.RegressionMetric):
    """Mean absolute error.

    Example:

    ::

        >>> from creme import metrics

        >>> y_true = [3, -0.5, 2, 7]
        >>> y_pred = [2.5, 0.0, 2, 8]

        >>> metric = metrics.MAE()
        >>> for y_t, y_p in zip(y_true, y_pred):
        ...     print(metric.update(y_t, y_p).get())
        0.5
        0.5
        0.333333...
        0.5

        >>> metric
        MAE: 0.5

    """

    def update(self, y_true, y_pred):
        return super().update(abs(y_true - y_pred))
