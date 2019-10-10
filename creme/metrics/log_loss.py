import math

from .. import stats

from . import base


__all__ = ['LogLoss']


class BaseLogLoss(base.BinaryMetric):

    @property
    def bigger_is_better(self):
        return False

    @property
    def requires_labels(self):
        return False


class LogLoss(stats.Mean, BaseLogLoss):
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

    def _get_log_loss(self, y_true, y_pred):
        p_true = y_pred.get(True, 0.) if isinstance(y_pred, dict) else y_pred
        p_true = self.clamp_proba(p_true)
        if y_true:
            return -math.log(p_true)
        return -math.log(1 - p_true)

    def update(self, y_true, y_pred, sample_weight=1.):
        ll = self._get_log_loss(y_true, y_pred)
        return super().update(x=ll, w=sample_weight)

    def revert(self, y_true, y_pred, sample_weight=1.):
        ll = self._get_log_loss(y_true, y_pred)
        return super().revert(x=ll, w=sample_weight)
