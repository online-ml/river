import numpy as np

from . import base


class AbsoluteLoss(base.RegressionLoss):
    """Computes the absolute loss, also known as the mean absolute error or L1 loss.

    Mathematically, it is defined as

    .. math:: L = |p_i - y_i|

    It's gradient w.r.t. to $p_i$ is

    .. math:: \\frac{\\partial L}{\\partial p_i} = sgn(p_i - y_i)

    Example:

        ::

            >>> from creme import optim

            >>> loss = optim.AbsoluteLoss()
            >>> loss(-42, 42)
            84
            >>> loss.gradient(1, 2)
            1
            >>> loss.gradient(2, 1)
            -1

    """

    def __call__(self, y_true, y_pred):
        return abs(y_pred - y_true)

    def gradient(self, y_true, y_pred):
        return np.sign(y_pred - y_true)
