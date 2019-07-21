from . import base


class HingeLoss(base.BinaryClassificationLoss):
    """Computes the hinge loss.

    Mathematically, it is defined as

    .. math:: L = max(0, 1 - p_i * y_i)

    It's gradient w.r.t. to $p_i$ is

    .. math::
        \\frac{\\partial L}{\\partial y_i} = \\left\{
        \\begin{array}{ll}
            \\ 0  &   p_iy_i \geqslant 1  \\\\
            \\ - y_i & p_iy_i < 1
        \\end{array}
        \\right.

    Example:

        ::

            >>> from creme import optim
            >>> import numpy as np
            >>> from sklearn import svm
            >>> from sklearn.metrics import hinge_loss

            >>> X = [[0], [1]]
            >>> y = [-1, 1]
            >>> lin_svm = svm.LinearSVC(random_state=0).fit(X, y)

            >>> y_true = [0, 1, 1]
            >>> y_pred = lin_svm.decision_function([[-2], [3], [0.5]])

            >>> hinge_loss([0, 1, 1], y_pred)
            0.303022...

            >>> loss = optim.HingeLoss()
            >>> np.mean([loss(y_t, y_p) for y_t, y_p in zip(y_true, y_pred)])
            0.303022...

    """

    def __call__(self, y_true, y_pred):
        # Our convention is to use 0s instead of -1s for negatives, but the Hinge loss uses -1s as
        # a convention
        y_true = y_true or -1
        return max(0, 1 - y_true * y_pred)

    def gradient(self, y_true, y_pred):
        """Returns the gradient with respect to ``y_pred``.

        References:
            1. `Wolfram Alpha <https://www.wolframalpha.com/input/?i=derivative+max(0,+1+-+p+*+y)+wrt+p>`_

        """
        y_true = y_true or -1
        if y_true * y_pred < 1:
            return -y_pred
        return 0


class EpsilonInsensitiveHingeLoss(base.RegressionLoss):

    def __init__(self, eps=0.1):
        self.eps = eps

    def __call__(self, y_true, y_pred):
        # Our convention is to use 0s instead of -1s for negatives, but the Hinge loss uses -1s as
        # a convention
        y_true = y_true or -1
        return max(0, abs(y_pred - y_true) - self.eps)

    def gradient(self, y_true, y_pred):
        """Returns the gradient with respect to ``y_pred``.

        References:
            1. `Wolfram Alpha <https://www.wolframalpha.com/input/?i=derivative+max(0,+abs(p+-+y)+-+eps)+wrt+p>`_

        """
        y_true = y_true or -1
        if y_pred > y_true + self.eps:
            return 1
        elif y_pred + self.eps < y_true:
            return -1
        return 0
