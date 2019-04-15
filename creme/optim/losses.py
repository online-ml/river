import abc
import math

import numpy as np


__all__ = [
    'AbsoluteLoss',
    'HingeLoss',
    'LogLoss',
    'SquaredLoss'
]


class Loss(abc.ABC):

    @abc.abstractmethod
    def __call__(self, y_true, y_pred) -> float:
        """Returns the loss."""

    @abc.abstractmethod
    def gradient(self, y_true, y_pred) -> float:
        """Returns the gradient with respect to ``y_pred``."""


class BinaryClassificationLoss(Loss):
    """Helper class used for organizing losses."""


class RegressionLoss(Loss):
    """Helper class used for organizing losses."""


class AbsoluteLoss(RegressionLoss):
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


class CauchyLoss(RegressionLoss):
    """Cauchy loss function.

    References:

        1. `Effect of MAE <https://www.kaggle.com/c/allstate-claims-severity/discussion/24520#140163>`_
        2. `Paris Madness <https://www.kaggle.com/raddar/paris-madness>`_

    """

    def __init__(self, C=80):
        self.C = C

    def __call__(self, y_true, y_pred):
        return abs(y_pred - y_true)

    def gradient(self, y_true, y_pred):
        diff = y_pred - y_true
        return diff / ((diff / self.C) ** 2 + 1)


class SquaredLoss(RegressionLoss):
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


class LogLoss(BinaryClassificationLoss):
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

    @staticmethod
    def _clip_proba(p):
        return max(min(p, 1 - 1e-15), 1e-15)

    def __call__(self, y_true, y_pred):
        y_pred = self._clip_proba(y_pred)
        if y_true:
            return -math.log(y_pred)
        return -math.log(1 - y_pred)

    def gradient(self, y_true, y_pred):
        return self._clip_proba(y_pred) - y_true


class HingeLoss(BinaryClassificationLoss):
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
            0.303036...

            >>> loss = optim.HingeLoss()
            >>> np.mean([loss(y_t, y_p) for y_t, y_p in zip(y_true, y_pred)])
            0.303036...

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


class EpsilonInsensitiveHingeLoss(RegressionLoss):

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
