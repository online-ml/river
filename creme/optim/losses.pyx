"""Loss functions."""
from libc cimport math


__all__ = [
    'Absolute',
    'Cauchy',
    'CrossEntropy',
    'Hinge',
    'EpsilonInsensitiveHinge',
    'Log',
    'Quantile',
    'Squared'
]


cdef double clamp_proba(double x):
    return math.fmax(math.fmin(x, 1 - 1e-15), 1e-15)


cdef class Loss:

    def __str__(self):
        return self.__class__.__name__


cdef class ClassificationLoss(Loss):
    pass


cdef class BinaryLoss(ClassificationLoss):
    """A loss appropriate for binary classification tasks."""

    cpdef double eval(self, bint y_true, double y_pred):
        """Returns the loss."""

    cpdef double gradient(self, bint y_true, double y_pred):
        """Returns the gradient with respect to ``y_pred``."""


cdef class MultiClassLoss(ClassificationLoss):
    """A loss appropriate for multi-class classification tasks."""

    cpdef double eval(self, object y_true, dict y_pred):
        """Returns the loss."""

    cpdef dict gradient(self, object y_true, dict y_pred):
        """Returns the gradient with respect to ``y_pred``."""


cdef class RegressionLoss(Loss):
    """A loss appropriate for regression tasks."""

    cpdef double eval(self, double y_true, double y_pred):
        """Returns the loss."""

    cpdef double gradient(self, double y_true, double y_pred):
        """Returns the gradient with respect to ``y_pred``."""


cdef class Absolute(RegressionLoss):
    """Absolute loss, also known as the mean absolute error or L1 loss.

    Mathematically, it is defined as

    .. math:: L = |p_i - y_i|

    It's gradient w.r.t. to $p_i$ is

    .. math:: \\frac{\\partial L}{\\partial p_i} = sgn(p_i - y_i)

    Example:

        ::

            >>> from creme import optim

            >>> loss = optim.losses.Absolute()
            >>> loss.eval(-42, 42)
            84.0
            >>> loss.gradient(1, 2)
            1.0
            >>> loss.gradient(2, 1)
            -1.0

    """

    cpdef double eval(self, double y_true, double y_pred):
        return abs(y_pred - y_true)

    cpdef double gradient(self, double y_true, double y_pred):
        if y_pred > y_true:
            return 1
        return -1


cdef class Cauchy(RegressionLoss):
    """Cauchy loss function.

    References:
        1. `Effect of MAE <https://www.kaggle.com/c/allstate-claims-severity/discussion/24520#140163>`_
        2. `Paris Madness <https://www.kaggle.com/raddar/paris-madness>`_

    """

    def __init__(self, C=80):
        self.C = C

    cpdef double eval(self, double y_true, double y_pred):
        return abs(y_pred - y_true)

    cpdef double gradient(self, double y_true, double y_pred):
        diff = y_pred - y_true
        return diff / ((diff / self.C) ** 2 + 1)


cdef class CrossEntropy(MultiClassLoss):
    """Cross entropy loss.

    This is a generalization of logistic loss to multiple classes.

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

            >>> loss = optim.losses.CrossEntropy()

            >>> for y_t, y_p in zip(y_true, y_pred):
            ...     print(loss.eval(y_t, y_p))
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

    cpdef double eval(self, object y_true, dict y_pred):
        cdef double total = 0

        for label, proba in y_pred.items():
            if y_true == label:
                total += math.log(clamp_proba(proba))

        return -total

    cpdef dict gradient(self, object y_true, dict y_pred):
        return {
            label: clamp_proba(y_pred.get(label, 0.)) - (y_true == label)
            for label in {*y_pred.keys(), y_true}
        }


cdef class Hinge(BinaryLoss):
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

            >>> loss = optim.losses.Hinge()
            >>> np.mean([loss.eval(y_t, y_p) for y_t, y_p in zip(y_true, y_pred)])
            0.303036...

    """

    cpdef double eval(self, bint y_true, double y_pred):
        # Our convention is to use 0s instead of -1s for negatives, but the Hinge loss uses -1s as
        # a convention
        y_true = y_true or -1
        return math.fmax(0, 1 - y_true * y_pred)

    cpdef double gradient(self, bint y_true, double y_pred):
        """Returns the gradient with respect to ``y_pred``.

        References:
            1. `Wolfram Alpha derivation <https://www.wolframalpha.com/input/?i=derivative+max(0,+1+-+p+*+y)+wrt+p>`_

        """
        y_true = y_true or -1
        if y_true * y_pred < 1:
            return -y_pred
        return 0


cdef class EpsilonInsensitiveHinge(RegressionLoss):

    cdef readonly double eps

    def __init__(self, eps=0.1):
        self.eps = eps

    cpdef double eval(self, double y_true, double y_pred):
        # Our convention is to use 0s instead of -1s for negatives, but the Hinge loss uses -1s as
        # a convention
        y_true = y_true or -1
        return math.fmax(0, math.fabs(y_pred - y_true) - self.eps)

    cpdef double gradient(self, double y_true, double y_pred):
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


cdef class Log(BinaryLoss):
    """Logarithmic loss.

    This loss function expects each provided ``y_pred`` to be a log odds ratio, not a probability.

    """

    cpdef double eval(self, bint y_true, double y_pred):
        y_true = y_true or -1
        z = y_pred * y_true
        if z > 18.:
            return math.exp(-z)
        if z < -18.:
            return -z
        return math.log(1.0 + math.exp(-z))

    cpdef double gradient(self, bint y_true, double y_pred):
        y_true = y_true or -1
        z = y_pred * y_true
        if z > 18.:
            return math.exp(-z) * -y_true
        if z < -18.:
            return -y_true
        return -y_true / (math.exp(z) + 1.0)


cdef class Quantile(RegressionLoss):
    """Quantile loss.

    Parameters:
        alpha (float): Desired quantile to attain.

    Example:

        ::

            >>> from creme import optim

            >>> loss = optim.losses.Quantile(0.5)
            >>> loss.eval(1, 3)
            1.0

            >>> loss.gradient(1, 3)
            0.5

            >>> loss.gradient(3, 1)
            -0.5

    References:
        1. `Wikipedia article on quantile regression <https://www.wikiwand.com/en/Quantile_regression>`_
        2. `Derivative from WolframAlpha <https://www.wolframalpha.com/input/?i=derivative+(y+-+p)+*+(alpha+-+Boole(y+-+p))+wrt+p>`_

    """

    cdef readonly double alpha

    def __init__(self, alpha):
        self.alpha = alpha

    cpdef double eval(self, double y_true, double y_pred):
        diff = y_pred - y_true
        return (self.alpha - (diff < 0)) * diff

    cpdef double gradient(self, double y_true, double y_pred):
        return (y_true < y_pred) - self.alpha


cdef class Squared(RegressionLoss):
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

            >>> loss = optim.losses.Squared()
            >>> loss.eval(-4, 5)
            81.0
            >>> loss.gradient(-4, 5)
            18.0
            >>> loss.gradient(5, -4)
            -18.0

    """

    cpdef double eval(self, double y_true, double y_pred):
        return (y_pred - y_true) * (y_pred - y_true)

    cpdef double gradient(self, double y_true, double y_pred):
        return 2. * (y_pred - y_true)
