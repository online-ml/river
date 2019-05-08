from .. import stats

from . import base


__all__ = ['MAE']


class MAE(stats.Mean, base.RegressionMetric):
    """Mean absolute error.

    Example:

        ::

            >>> import math
            >>> from creme import metrics
            >>> from sklearn.metrics import mean_absolute_error

            >>> y_true = [3, -0.5, 2, 7]
            >>> y_pred = [2.5, 0.0, 2, 8]

            >>> metric = metrics.MAE()
            >>> for i, (y_t, y_p) in enumerate(zip(y_true, y_pred)):
            ...     metric = metric.update(y_t, y_p)
            ...     assert math.isclose(
            ...         metric.get(),
            ...         mean_absolute_error(y_true[:i+1], y_pred[:i+1])
            ...     )

            >>> metric
            MAE: 0.5

    """

    @property
    def bigger_is_better(self):
        return False

    def update(self, y_true, y_pred):
        return super().update(abs(y_true - y_pred))
