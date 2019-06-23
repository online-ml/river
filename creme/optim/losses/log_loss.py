import math

from . import base


class LogLoss(base.BinaryClassificationLoss):
    """Computes the logarithmic loss.

    Mathematically, it is defined as

    .. math:: L = -y_i log(p_i) + (1-y_i) log(1-p_i)

    It's gradient w.r.t. to $p_i$ is

    .. math:: \\frac{\\partial L}{\\partial p_i} = sign(p_i - y_i)

    Example:

        ::

            >>> from creme import optim

            >>> loss = optim.LogLoss()
            >>> loss(1, 0.5)
            0.693147...
            >>> loss(1, 0)
            34.53877...
            >>> loss.gradient(1, 0.2)
            -0.8
            >>> loss.gradient(0, 0.2)
            0.2

    """

    def __call__(self, y_true, y_pred):
        y_pred = self.clamp_proba(y_pred)
        if y_true:
            return -math.log(y_pred)
        return -math.log(1 - y_pred)

    def gradient(self, y_true, y_pred):
        return self.clamp_proba(y_pred) - y_true
