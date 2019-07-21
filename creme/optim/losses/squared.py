from . import base


class SquaredLoss(base.RegressionLoss):
    """Computes the squared loss, also known as the L2 loss.

    Mathematically, it is defined as

    .. math:: L = (p_i - y_i) ^ 2

    It's gradient w.r.t. to $p_i$ is

    .. math:: \\frac{\\partial L}{\\partial p_i} = 2 \times (p_i - y_i)

    One thing to note is that this convention is consistent with Vowpal Wabbit and PyTorch, but
    not with scikit-learn. Indeed scikit-learn divides the loss by 2, making the 2 dissapear in
    the gradient.

    Example:

        ::

            >>> from creme import optim

            >>> loss = optim.SquaredLoss()
            >>> loss(-4, 5)
            81
            >>> loss.gradient(-4, 5)
            18.0
            >>> loss.gradient(5, -4)
            -18.0

    """

    def __call__(self, y_true, y_pred):
        return (y_pred - y_true) * (y_pred - y_true)

    def gradient(self, y_true, y_pred):
        return 2. * (y_pred - y_true)
