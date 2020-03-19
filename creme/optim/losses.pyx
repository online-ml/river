"""Loss functions."""
from libc cimport math
from .. import utils


__all__ = [
    'Absolute',
    'BinaryFocalLoss',
    'Cauchy',
    'CrossEntropy',
    'Hinge',
    'EpsilonInsensitiveHinge',
    'Log',
    'Perceptron',
    'Poisson',
    'Quantile',
    'Squared'
]


cdef double clamp_proba(double x):
    return math.fmax(math.fmin(x, 1 - 1e-15), 1e-15)


cdef class Loss:

    def __str__(self):
        return utils.pretty.format_object(self)

    def mean_func(self, y_pred):
        """Mean function.

        This is the inverse of the link function.

        References:
            1. `Wikipedia section on link and mean function <https://www.wikiwand.com/en/Generalized_linear_model#/Link_function>`_

        """


cdef class ClassificationLoss(Loss):
    pass


cdef class BinaryLoss(ClassificationLoss):
    """A loss appropriate for binary classification tasks."""

    cpdef double eval(self, bint y_true, double y_pred):
        """Returns the loss."""

    cpdef double gradient(self, bint y_true, double y_pred):
        """Returns the gradient with respect to ``y_pred``."""

    cpdef double mean_func(self, double y_pred):
        return utils.math.sigmoid(y_pred)


cdef class MultiClassLoss(ClassificationLoss):
    """A loss appropriate for multi-class classification tasks."""

    cpdef double eval(self, object y_true, dict y_pred):
        """Returns the loss."""

    cpdef dict gradient(self, object y_true, dict y_pred):
        """Returns the gradient with respect to ``y_pred``."""

    cpdef dict mean_func(self, dict y_pred):
        return utils.math.softmax(y_pred)


cdef class RegressionLoss(Loss):
    """A loss appropriate for regression tasks."""

    cpdef double eval(self, double y_true, double y_pred):
        """Returns the loss."""

    cpdef double gradient(self, double y_true, double y_pred):
        """Returns the gradient with respect to ``y_pred``."""

    cpdef double mean_func(self, double y_pred):
        return y_pred


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
        1. `"Effect of MAE" Kaggle discussion <https://www.kaggle.com/c/allstate-claims-severity/discussion/24520#140163>`_
        2. `Paris Madness Kaggle kernel <https://www.kaggle.com/raddar/paris-madness>`_

    """

    def __init__(self, C=80):
        self.C = C

    cpdef double eval(self, double y_true, double y_pred):
        return math.fabs(y_pred - y_true)

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
            1.222454
            1.116929
            1.437209
            1.509797

            >>> for y_t, y_p in zip(y_true, y_pred):
            ...     print(loss.gradient(y_t, y_p))
            {0: -0.70549363, 1: 0.34216758, 2: 0.36332605}
            {0: 0.21290077, 1: -0.67271668, 2: 0.45981591}
            {0: 0.42860913, 1: 0.33380113, 2: -0.76241026}
            {0: 0.44941979, 1: 0.32962558, 2: -0.77904537}

    References:
        1. `What is Softmax regression and how is it related to Logistic regression? <https://github.com/rasbt/python-machine-learning-book/blob/master/faq/softmax_regression.md>`_

    """

    cdef readonly dict class_weight

    def __init__(self, class_weight=None):
        if class_weight is None:
            class_weight = {}
        self.class_weight = class_weight

    cpdef double eval(self, object y_true, dict y_pred):
        cdef double total = 0

        for label, proba in y_pred.items():
            if y_true == label:
                total += self.class_weight.get(label, 1.) * math.log(clamp_proba(proba))

        return -total

    cpdef dict gradient(self, object y_true, dict y_pred):
        return {
            label: (
                self.class_weight.get(label, 1.) *
                (clamp_proba(y_pred.get(label, 0.)) - (y_true == label))
            )
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

    Parameters:
        threshold (float): Margin threshold. 1 yield the loss used in SVMs, whilst 0 is equivalent to
            the loss used in the Perceptron algorithm.

    Example:

        ::

            >>> from creme import optim

            >>> loss = optim.losses.Hinge(threshold=1)
            >>> loss.eval(1, .2)
            0.8

            >>> loss.gradient(1, .2)
            -0.2

    """

    cdef readonly double threshold

    def __init__(self, double threshold=1.):
        self.threshold = threshold

    cpdef double eval(self, bint y_true, double y_pred):
        # Convert 0 to -1
        y_true = y_true or -1
        return math.fmax(0, self.threshold - y_true * y_pred)

    cpdef double gradient(self, bint y_true, double y_pred):
        """Returns the gradient with respect to ``y_pred``.

        References:
            1. `WolframAlpha derivation <https://www.wolframalpha.com/input/?i=derivative+max(0,+1+-+p+*+y)+wrt+p>`_

        """
        y_true = y_true or -1
        if y_true * y_pred < self.threshold:
            return -y_pred
        return 0


cdef class EpsilonInsensitiveHinge(RegressionLoss):

    cdef readonly double eps

    def __init__(self, eps=0.1):
        self.eps = eps

    cpdef double eval(self, double y_true, double y_pred):
        # Convert 0 to -1
        y_true = y_true or -1
        return math.fmax(0, math.fabs(y_pred - y_true) - self.eps)

    cpdef double gradient(self, double y_true, double y_pred):
        """Returns the gradient with respect to ``y_pred``.

        References:
            1. `WolframAlpha derivation <https://www.wolframalpha.com/input/?i=derivative+max(0,+abs(p+-+y)+-+eps)+wrt+p>`_

        """
        y_true = y_true or -1
        if y_pred > y_true + self.eps:
            return 1
        elif y_pred + self.eps < y_true:
            return -1
        return 0


cdef class Log(BinaryLoss):
    """Logarithmic loss.

    This loss function expects each provided ``y_pred`` to be a logit. In other words if must be
    the raw output of a linear model or a neural network.

    References:
        1. `Logit Wikipedia page <https://www.wikiwand.com/en/Logit>`_

    """

    cdef readonly double weight_pos
    cdef readonly double weight_neg

    def __init__(self, weight_pos=1., weight_neg=1.):
        self.weight_pos = weight_pos
        self.weight_neg = weight_neg

    cpdef double eval(self, bint y_true, double y_pred):
        weight = self.weight_pos
        if y_true == 0:
            y_true = -1
            weight = self.weight_neg

        z = y_pred * y_true
        if z > 18.:
            return weight * math.exp(-z)
        if z < -18.:
            return weight * -z
        return weight * math.log(1. + math.exp(-z))

    cpdef double gradient(self, bint y_true, double y_pred):
        weight = self.weight_pos
        if y_true == 0:
            y_true = -1
            weight = self.weight_neg

        z = y_pred * y_true
        if z > 18.:
            return weight * math.exp(-z) * -y_true
        if z < -18.:
            return weight * -y_true
        return weight * -y_true / (math.exp(z) + 1.)


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
    """Squared loss, also known as the L2 loss.

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


class BinaryFocalLoss(BinaryLoss):
    """Binary focal loss.

    This implements the "star" algorithm from the appendix of the focal loss paper.

    Parameters:
        gamma (float)
        beta (float)

    Refenrences:
        1. `Lin, T.Y., Goyal, P., Girshick, R., He, K. and Dollár, P., 2017. Focal loss for dense object detection. In Proceedings of the IEEE international conference on computer vision (pp. 2980-2988). <https://arxiv.org/pdf/1708.02002.pdf>`_

    """

    def __init__(self, gamma=2, beta=1):
        self.gamma = gamma
        self.beta = beta

    def eval(self, y_true, y_pred):

        # Convert 0 to -1
        y_true = int(y_true or -1)

        xt = y_true * y_pred
        pt = utils.math.sigmoid(self.gamma * xt + self.beta)

        return -math.log(pt) / self.gamma

    def gradient(self, y_true, y_pred):

        # Convert 0 to -1
        y_true = int(y_true or -1)

        xt = y_true * y_pred
        pt = utils.math.sigmoid(self.gamma * xt + self.beta)

        return y_true * (pt - 1)


cdef class Poisson(RegressionLoss):
    """Poisson loss.

    The Poisson loss is usually more suited for regression with count data than the squared loss.

    Mathematically, it is defined as

    .. math:: L = exp(p_i) - y_i \\times p_i

    It's gradient w.r.t. to $p_i$ is

    .. math:: \\frac{\\partial L}{\\partial p_i} = exp(p_i) - y_i

    """

    cpdef double eval(self, double y_true, double y_pred):
        return math.exp(y_pred) - y_true * y_pred

    cpdef double gradient(self, double y_true, double y_pred):
        return math.exp(y_pred) - y_true

    cpdef double mean_func(self, double y_pred):
        return math.exp(y_pred)


cdef class Perceptron(Hinge):
    """Perceptron loss.

    The Perceptron loss is the loss used in the Perceptron algorithm. Using this loss in a logistic
    regression yields the Perceptron algorithm.

    Mathematically, it is defined as

    .. math:: L = exp(p_i) - y_i \\times p_i

    It's gradient w.r.t. to $p_i$ is

    .. math:: \\frac{\\partial L}{\\partial p_i} = exp(p_i) - y_i

    References:
        1. `Wikipedia page on the Perceptron algorithm <https://www.wikiwand.com/en/Perceptron>`_

    """

    def __init__(self):
        super().__init__(threshold=0.)
