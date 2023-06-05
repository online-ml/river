from __future__ import annotations

import math

from river import metrics

__all__ = ["LogLoss"]


class LogLoss(metrics.base.MeanMetric, metrics.base.BinaryMetric):
    """Binary logarithmic loss.

    Examples
    --------

    >>> from river import metrics

    >>> y_true = [True, False, False, True]
    >>> y_pred = [0.9,  0.1,   0.2,   0.65]

    >>> metric = metrics.LogLoss()
    >>> for yt, yp in zip(y_true, y_pred):
    ...     metric = metric.update(yt, yp)
    ...     print(metric.get())
    0.105360
    0.105360
    0.144621
    0.216161

    >>> metric
    LogLoss: 0.216162

    """

    _fmt = ""

    @property
    def bigger_is_better(self):
        return False

    @property
    def requires_labels(self):
        return False

    def _eval(self, y_true, y_pred):
        p_true = y_pred.get(True, 0.0) if isinstance(y_pred, dict) else y_pred
        p_true = self._clamp_proba(p_true)
        if y_true:
            return -math.log(p_true)
        return -math.log(1 - p_true)
