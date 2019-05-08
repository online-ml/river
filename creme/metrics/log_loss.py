from .. import optim
from .. import stats

from . import base


__all__ = ['LogLoss']


class LogLoss(stats.Mean, base.BinaryClassificationMetric):
    """Binary logarithmic loss.

    Example:

        ::

            >>> from creme import metrics
            >>> from sklearn.metrics import log_loss

            >>> y_true = [True, False, False, True]
            >>> y_pred = [0.9,  0.1,   0.2,   0.65]

            >>> metric = metrics.LogLoss()
            >>> for y_t, y_p in zip(y_true, y_pred):
            ...     metric = metric.update(y_t, y_p)
            ...     print(metric.get())
            0.105360...
            0.105360...
            0.144621...
            0.216161...

            >>> metric
            LogLoss: 0.216162

    """

    @property
    def bigger_is_better(self):
        return False

    @property
    def requires_labels(self):
        return False

    def update(self, y_true, y_pred):
        ll = optim.LogLoss().__call__
        return super().update(ll(y_true, y_pred[True] if isinstance(y_pred, dict) else y_pred))
