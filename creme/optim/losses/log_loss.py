import math

from . import base


class LogLoss(base.BinaryClassificationLoss):
    """Logarithmic loss."""

    def __call__(self, y_true, y_pred):
        y_true = float(y_true or -1)
        z = y_pred * y_true
        if z > 18.:
            return math.exp(-z)
        if z < -18.:
            return -z
        return math.log(1.0 + math.exp(-z))

    def gradient(self, y_true, y_pred):
        y_true = float(y_true or -1)
        z = y_pred * y_true
        if z > 18.:
            return math.exp(-z) * -y_true
        if z < -18.:
            return -y_true
        return -y_true / (math.exp(z) + 1.0)
