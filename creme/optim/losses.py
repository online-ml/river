import abc
import math

import numpy as np


__all__ = ['SquaredLoss', 'LogLoss', 'AbsoluteLoss', 'HingeLoss']


class Loss(abc.ABC):

    @abc.abstractmethod
    def __call__(self, y_true, y_pred) -> float:
        """Returns the loss."""
        pass

    @abc.abstractmethod
    def gradient(self, y_true, y_pred) -> float:
        """Returns the gradient with respect to `y_pred`."""
        pass


class AbsoluteLoss(Loss):
    """Computes the absolute loss, also known as the mean absolute error or L1 loss.

    Mathematically, it is defined as

    .. math:: L = |p_i - y_i|

    It's gradient w.r.t. to $p_i$ is

    .. math:: \\frac{\\partial L}{\\partial p_i} = sgn(p_i - y_i)

    """

    def __call__(self, y_true, y_pred):
        return abs(y_pred - y_true)

    def gradient(self, y_true, y_pred):
        return np.sign(y_pred - y_true)


class SquaredLoss(Loss):
    """Computes the squared loss, also known as the L2 loss.

    Mathematically, it is defined as

    .. math:: L = (p_i - y_i)^2

    It's gradient w.r.t. to $p_i$ is

    .. math:: \\frac{\\partial L}{\\partial p_i} = 2(p_i - y_i)

    """

    def __call__(self, y_true, y_pred):
        return (y_pred - y_true) ** 2

    def gradient(self, y_true, y_pred):
        return 2 * (y_pred - y_true)


class LogLoss(Loss):
    """Computes the logarithmic loss.

    Mathematically, it is defined as

    .. math:: L = -y_i log(p_i) + (1-y_i) log(1-p_i)

    It's gradient w.r.t. to $p_i$ is

    .. math:: \\frac{\\partial L}{\\partial p_i} = sign(p_i - y_i)

    """

    @staticmethod
    def _clip_proba(p):
        return max(min(p, 1 - 1e-15), 1e-15)

    def __call__(self, y_true, y_pred):
        y_pred = self._clip_proba(y_pred)
        return -(y_true * math.log(y_pred) + (1 - y_true) * math.log(1 - y_pred))

    def gradient(self, y_true, y_pred):
        return self._clip_proba(y_pred) - y_true


class HingeLoss(Loss):
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

        >>> import numpy as np
        >>> from sklearn import svm
        >>> from sklearn.metrics import hinge_loss
        >>> import creme
        >>> X = [[0], [1]]
        >>> y = [-1, 1]
        >>> est = svm.LinearSVC(random_state=0)
        >>> est.fit(X, y)
        LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
             intercept_scaling=1, loss='squared_hinge', max_iter=1000,
             multi_class='ovr', penalty='l2', random_state=0, tol=0.0001,
             verbose=0)

        >>> y_true = [0, 1, 1]
        >>> pred_decision = est.decision_function([[-2], [3], [0.5]])

        >>> hinge_loss([0, 1, 1], pred_decision)
        0.303036...

        >>> loss = creme.optim.HingeLoss()
        >>> np.mean([loss(y_t,pred) for y_t, pred in zip(y_true, pred_decision)])
        0.303036...

    """

    def __call__(self, y_true, y_pred):
        # Our convention is to use 0s instead of -1s for negatives, but the Hinge loss uses -1s as
        # a convention
        y_true = y_true or -1
        return max(0, 1 - y_true * y_pred)

    def gradient(self, y_true, y_pred):
        """Returns the gradient with respect to `y_pred`.

        `Wolfram Alpha <https://www.wolframalpha.com/input/?i=derivative+max(0,+1+-+p+*+y)+wrt+p>`_

        """
        y_true = y_true or -1
        if y_true * y_pred < 1:
            return -y_pred
        return 0


class EpsilonInsensitiveHingeLoss(Loss):

    def __init__(self, eps=0.1):
        self.eps = eps

    def __call__(self, y_true, y_pred):
        # Our convention is to use 0s instead of -1s for negatives, but the Hinge loss uses -1s as
        # a convention
        y_true = y_true or -1
        return max(0, abs(y_pred - y_true) - self.eps)

    def gradient(self, y_true, y_pred):
        """Returns the gradient with respect to `y_pred`.

        `Wolfram Alpha <https://www.wolframalpha.com/input/?i=derivative+max(0,+abs(p+-+y)+-+eps)+wrt+p>`_

        """
        y_true = y_true or -1
        if y_pred > y_true + self.eps:
            return 1
        elif y_pred + self.eps < y_true:
            return -1
        return 0
