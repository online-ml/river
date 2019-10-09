from .. import stats

from . import base


__all__ = ['SMAPE']


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

    def update(self, y_true, y_pred, sample_weight=1.):
        return super().update(
            x=abs(y_true - y_pred) / (abs(y_true) + abs(y_pred)),
            w=sample_weight
        )

    def get(self):
        return 100 * super().get()
