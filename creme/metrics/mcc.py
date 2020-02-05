import math

from . import base
from . import confusion


__all__ = ['MCC']


class MCC(base.BinaryMetric):
    """Matthews correlation coefficient.

    References:
        1. `Wikipedia article <https://www.wikiwand.com/en/Matthews_correlation_coefficient>`_

    """

    def __init__(self):
        self.cm = confusion.ConfusionMatrix()

    @property
    def bigger_is_better(self):
        return True

    @property
    def requires_labels(self):
        return True

    def update(self, y_true, y_pred, sample_weight=1.):
        self.cm.update(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)
        return self

    def revert(self, y_true, y_pred, sample_weight=1.):
        self.cm.revert(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)
        return self

    def get(self):
        tp = self.cm.counts.get(True, {}).get(True, 0)
        tn = self.cm.counts.get(False, {}).get(False, 0)
        fp = self.cm.counts.get(False, {}).get(True, 0)
        fn = self.cm.counts.get(True, {}).get(False, 0)

        n = (tp + tn + fp + fn) or 1
        s = (tp + fn) / n
        p = (tp + fp) / n

        try:
            return (tp / n - s * p) / math.sqrt(p * s * (1 - s) * (1 - p))
        except ZeroDivisionError:
            return 0.
