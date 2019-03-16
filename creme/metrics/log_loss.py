from .. import optim
from .. import stats

from . import base


class LogLoss(stats.Mean, base.BinaryClassificationMetric):
    """

    Example:

        >>> from creme import metrics

        >>> y_true = [True, False, False, True]
        >>> y_pred = [0.9,  0.1,   0.2,   0.65]

        >>> metric = metrics.LogLoss()
        >>> for y_t, y_p in zip(y_true, y_pred):
        ...     print(metric.update(y_t, y_p).get())
        0.105360...
        0.105360...
        0.144621...
        0.216161...

        >>> metric
        LogLoss: 0.216162

    """

    ll = optim.LogLoss().__call__

    @property
    def requires_labels(self):
        return False

    def update(self, y_true, y_pred):
        return super().update(self.ll(y_true, y_pred[True] if isinstance(y_pred, dict) else y_pred))
