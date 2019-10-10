from .. import stats

from . import base


__all__ = ['MAE']


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

    def update(self, y_true, y_pred, sample_weight=1.):
        return super().update(x=abs(y_true - y_pred), w=sample_weight)

    def revert(self, y_true, y_pred, sample_weight=1.):
        return super().revert(x=abs(y_true - y_pred), w=sample_weight)
