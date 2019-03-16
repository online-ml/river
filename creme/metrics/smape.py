from .. import stats

from . import base


class SMAPE(stats.Mean, base.RegressionMetric):
    """Symmetric mean absolute percentage error.

    Example:

    ::

        >>> from creme import metrics

        >>> y_true = [100, 100]
        >>> y_pred = [110, 90]

        >>> metric = metrics.SMAPE()
        >>> for y_t, y_p in zip(y_true, y_pred):
        ...     print(metric.update(y_t, y_p))
        SMAPE: 4.761905
        SMAPE: 5.012531

        >>> metric
        SMAPE: 5.012531

    """

    def update(self, y_true, y_pred):
        return super().update(abs(y_true - y_pred) / (abs(y_true) + abs(y_pred)))

    def get(self):
        return 100 * super().get()
