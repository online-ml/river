from __future__ import annotations

import math

from river import metrics

__all__ = ["MCC"]


class MCC(metrics.base.BinaryMetric):
    """Matthews correlation coefficient.

    Parameters
    ----------
    cm
        This parameter allows sharing the same confusion matrix between multiple metrics. Sharing a
        confusion matrix reduces the amount of storage and computation time.
    pos_val
        Value to treat as "positive".

    Examples
    --------

    >>> from river import metrics

    >>> y_true = [True, True, True, False]
    >>> y_pred = [True, False, True, True]

    >>> mcc = metrics.MCC()

    >>> for yt, yp in zip(y_true, y_pred):
    ...     mcc = mcc.update(yt, yp)

    >>> mcc
    MCC: -0.333333

    References
    ----------
    [^1]: [Wikipedia article](https://www.wikiwand.com/en/Matthews_correlation_coefficient)

    """

    _fmt = ""

    def get(self):
        tp = self.cm.true_positives(self.pos_val)
        tn = self.cm.true_negatives(self.pos_val)
        fp = self.cm.false_positives(self.pos_val)
        fn = self.cm.false_negatives(self.pos_val)

        n = (tp + tn + fp + fn) or 1
        s = (tp + fn) / n
        p = (tp + fp) / n

        try:
            return (tp / n - s * p) / math.sqrt(p * s * (1 - s) * (1 - p))
        except ZeroDivisionError:
            return 0.0
