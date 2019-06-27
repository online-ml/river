from . import base


class SquaredLoss(base.RegressionLoss):
    """Computes the squared loss, also known as the L2 loss.

    Mathematically, it is defined as

    .. math:: L = \\frac{(p_i - y_i)^2}{2}

    It's gradient w.r.t. to $p_i$ is

    .. math:: \\frac{\\partial L}{\\partial p_i} = p_i - y_i

    Example:

        ::

            >>> from creme import optim

            >>> loss = optim.SquaredLoss()
            >>> loss(-4, 5)
            40.5
            >>> loss.gradient(1, 4)
            3
            >>> loss.gradient(4, 1)
            -3

    """

    def __call__(self, y_true, y_pred):
        return .5 * (y_pred - y_true) * (y_pred - y_true)

    def gradient(self, y_true, y_pred):
        return y_pred - y_true
