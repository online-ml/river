import abc
import math

import numpy as np

from .. import utils


__all__ = [
    'AbsoluteLoss',
    'BinaryClassificationLoss',
    'CauchyLoss',
    'CrossEntropy',
    'HingeLoss',
    'LogLoss',
    'MultiClassificationLoss',
    'RegressionLoss',
    'SquaredLoss'
]


def clip_proba(p):
    return utils.clip(p, minimum=1e-15, maximum=1 - 1e-15)


class Loss(abc.ABC):

    @abc.abstractmethod
    def __call__(self, y_true, y_pred) -> float:
        """Returns the loss."""

    @abc.abstractmethod
    def gradient(self, y_true, y_pred) -> float:
        """Returns the gradient with respect to ``y_pred``."""


class BinaryClassificationLoss(Loss):
    """A loss appropriate binary classification tasks."""


class MultiClassificationLoss(Loss):
    """A loss appropriate for multi-class classification tasks."""


class RegressionLoss(Loss):
    """A loss appropriate for regression tasks."""


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

    def __call__(self, y_true, y_pred):
        y_pred = clip_proba(y_pred)
        if y_true:
            return -math.log(y_pred)
        return -math.log(1 - y_pred)

    def gradient(self, y_true, y_pred):
        return clip_proba(y_pred) - y_true


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


class CrossEntropy(MultiClassificationLoss):
    """Cross entropy is a generalization of logistic loss to multiple classes.

    Example:

        ::

            >>> from creme import optim

            >>> y_true = [0, 1, 2, 2]
            >>> y_pred = [
            ...     {0: 0.29450637, 1: 0.34216758, 2: 0.36332605},
            ...     {0: 0.21290077, 1: 0.32728332, 2: 0.45981591},
            ...     {0: 0.42860913, 1: 0.33380113, 2: 0.23758974},
            ...     {0: 0.44941979, 1: 0.32962558, 2: 0.22095463}
            ... ]

            >>> loss = optim.CrossEntropy()

            >>> for y_t, y_p in zip(y_true, y_pred):
            ...     print(loss(y_t, y_p))
            1.222454...
            1.116929...
            1.437209...
            1.509797...

            >>> for y_t, y_p in zip(y_true, y_pred):
            ...     print(loss.gradient(y_t, y_p))
            {0: -0.70549363, 1: 0.34216758, 2: 0.36332605}
            {0: 0.21290077, 1: -0.67271668, 2: 0.45981591}
            {0: 0.42860913, 1: 0.33380113, 2: -0.76241026}
            {0: 0.44941979, 1: 0.32962558, 2: -0.77904537}

    References:

        1. `What is Softmax regression and how is it related to Logistic regression? <https://github.com/rasbt/python-machine-learning-book/blob/master/faq/softmax_regression.md>`_

    """

    def __call__(self, y_true, y_pred):
        return -sum(
            (y_true == label) * math.log(clip_proba(proba))
            for label, proba in y_pred.items()
        )

    def gradient(self, y_true, y_pred):
        return {
            label: clip_proba(y_pred.get(label, 0.)) - (y_true == label)
            for label in {*y_pred.keys(), y_true}
        }
