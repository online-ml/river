from .. import stats

from . import base


__all__ = ['SMAPE']


class SMAPE(stats.Mean, base.RegressionMetric):
    """Symmetric mean absolute percentage error.

    Example:

        ::

            >>> from creme import metrics

            >>> y_true = [0.07533, 0.07533, 0.07533, 0.07533, 0.07533, 0.07533, 0.0672, 0.0672]
            >>> y_pred = [0.102, 0.107, 0.047, 0.1, 0.032, 0.047, 0.108, 0.089]

            >>> metric = metrics.SMAPE()
            >>> for y_t, y_p in zip(y_true, y_pred):
            ...     metric = metric.update(y_t, y_p)

            >>> metric
            SMAPE: 42.603066

    """

    def update(self, y_true, y_pred, sample_weight=1.):
        return super().update(
            x=2. * abs(y_true - y_pred) / (abs(y_true) + abs(y_pred)),
            w=sample_weight
        )

    def get(self):
        return 100 * super().get()
