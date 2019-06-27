import math

from . import base
from . import confusion


__all__ = ['MCC', 'RollingMCC']


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

    def update(self, y_true, y_pred):
        self.cm.update(y_true=y_true, y_pred=y_pred)
        return self

    def get(self):
        tp = self.cm.get(True, {}).get(True, 0)
        tn = self.cm.get(False, {}).get(False, 0)
        fp = self.cm.get(False, {}).get(True, 0)
        fn = self.cm.get(True, {}).get(False, 0)

        n = tp + tn + fp + fn
        s = (tp + fn) / n
        p = (tp + fp) / n

        try:
            return (tp / n - s * p) / math.sqrt(p * s * (1 - s) * (1 - p))
        except ZeroDivisionError:
            return 0.


class RollingMCC(MCC):
    """Matthews correlation coefficient.

    Parameters:
        window_size (int): Size of the window of recent values to consider.

    References:
        1. `Wikipedia article <https://www.wikiwand.com/en/Matthews_correlation_coefficient>`_

    """

    def __init__(self, window_size):
        self.cm = confusion.RollingConfusionMatrix(window_size=window_size)

    @property
    def window_size(self):
        return self.cm.window_size
