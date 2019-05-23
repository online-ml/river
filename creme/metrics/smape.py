from .. import stats

from . import base


__all__ = ['RollingSMAPE', 'SMAPE']


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

    @property
    def bigger_is_better(self):
        return False

    def update(self, y_true, y_pred):
        return super().update(abs(y_true - y_pred) / (abs(y_true) + abs(y_pred)))

    def get(self):
        return 100 * super().get()


class RollingSMAPE(stats.RollingMean, base.RegressionMetric):
    """Rolling symmetric mean absolute percentage error.

    Example:

        ::

            >>> from creme import metrics

            >>> y_true = [100, 100, 100]
            >>> y_pred = [110, 90, 80]

            >>> metric = metrics.RollingSMAPE(window_size=2)
            >>> for y_t, y_p in zip(y_true, y_pred):
            ...     print(metric.update(y_t, y_p))
            RollingSMAPE: 4.761905
            RollingSMAPE: 5.012531
            RollingSMAPE: 8.187135

    """

    @property
    def bigger_is_better(self):
        return False

    def update(self, y_true, y_pred):
        return super().update(abs(y_true - y_pred) / (abs(y_true) + abs(y_pred)))

    def get(self):
        return 100 * super().get()
