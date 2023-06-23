"""Loss functions.

Each loss function is intended to work with both single values as well as numpy vectors.

"""
from __future__ import annotations

import math

import numpy as np
from scipy import special  # type: ignore

from river import base, utils
from river.optim.base import Loss

__all__ = [
    "Absolute",
    "BinaryLoss",
    "BinaryFocalLoss",
    "Cauchy",
    "CrossEntropy",
    "Hinge",
    "Huber",
    "EpsilonInsensitiveHinge",
    "Log",
    "MultiClassLoss",
    "Poisson",
    "Quantile",
    "RegressionLoss",
    "Squared",
]


def clamp_proba(x: float) -> float:
    return utils.math.clamp(x=x, minimum=1e-15, maximum=1 - 1e-15)


class BinaryLoss(Loss):
    """A loss appropriate for binary classification tasks."""

    def mean_func(self, y_pred):
        if isinstance(y_pred, np.ndarray):
            return 1.0 / (1.0 + np.exp(-y_pred))
        return utils.math.sigmoid(y_pred)


class MultiClassLoss(Loss):
    """A loss appropriate for multi-class classification tasks."""

    def mean_func(self, y_pred):
        if isinstance(y_pred, np.ndarray):
            return special.softmax(y_pred)
        return utils.math.softmax(y_pred)


class RegressionLoss(Loss):
    """A loss appropriate for regression tasks."""

    def mean_func(self, y_pred):
        return y_pred


class Absolute(RegressionLoss):
    """Absolute loss, also known as the mean absolute error or L1 loss.

    Mathematically, it is defined as

    $$L = |p_i - y_i|$$

    It's gradient w.r.t. to $p_i$ is

    $$\\frac{\\partial L}{\\partial p_i} = sgn(p_i - y_i)$$

    Examples
    --------

    >>> from river import optim

    >>> loss = optim.losses.Absolute()
    >>> loss(-42, 42)
    84
    >>> loss.gradient(1, 2)
    1
    >>> loss.gradient(2, 1)
    -1

    """

    def __call__(self, y_true, y_pred):
        if isinstance(y_true, np.ndarray):
            return np.abs(y_pred - y_true)
        return abs(y_pred - y_true)

    def gradient(self, y_true, y_pred):
        if isinstance(y_true, np.ndarray):
            return np.where(y_pred > y_true, 1, -1)

        if y_pred > y_true:
            return 1
        return -1


class Cauchy(RegressionLoss):
    """Cauchy loss function.

    Parameters
    ----------
    C

    References
    ----------
    [^1]: ["Effect of MAE" Kaggle discussion](https://www.kaggle.com/c/allstate-claims-severity/discussion/24520#140163)
    [^2]: [Paris Madness Kaggle kernel](https://www.kaggle.com/raddar/paris-madness)

    """

    def __init__(self, C=80):
        self.C = C

    def __call__(self, y_true, y_pred):
        if isinstance(y_true, np.ndarray):
            return np.abs(y_pred - y_true)
        return abs(y_pred - y_true)

    def gradient(self, y_true, y_pred):
        diff = y_pred - y_true
        return diff / ((diff / self.C) ** 2 + 1)


class CrossEntropy(MultiClassLoss):
    """Cross entropy loss.

    This is a generalization of logistic loss to multiple classes.

    Parameters
    ----------
    class_weight
        A dictionary that indicates what weight to associate with each class.

    Examples
    --------

    >>> from river import optim

    >>> y_true = [0, 1, 2, 2]
    >>> y_pred = [
    ...     {0: 0.29450637, 1: 0.34216758, 2: 0.36332605},
    ...     {0: 0.21290077, 1: 0.32728332, 2: 0.45981591},
    ...     {0: 0.42860913, 1: 0.33380113, 2: 0.23758974},
    ...     {0: 0.44941979, 1: 0.32962558, 2: 0.22095463}
    ... ]

    >>> loss = optim.losses.CrossEntropy()

    >>> for yt, yp in zip(y_true, y_pred):
    ...     print(loss(yt, yp))
    1.222454
    1.116929
    1.437209
    1.509797

    >>> for yt, yp in zip(y_true, y_pred):
    ...     print(loss.gradient(yt, yp))
    {0: -0.70549363, 1: 0.34216758, 2: 0.36332605}
    {0: 0.21290077, 1: -0.67271668, 2: 0.45981591}
    {0: 0.42860913, 1: 0.33380113, 2: -0.76241026}
    {0: 0.44941979, 1: 0.32962558, 2: -0.77904537}

    References
    ----------
    [^1]: [What is Softmax regression and how is it related to Logistic regression?](https://github.com/rasbt/python-machine-learning-book/blob/master/faq/softmax_regression.md)

    """

    def __init__(self, class_weight: dict[base.typing.ClfTarget, float] | None = None):
        if class_weight is None:
            class_weight = {}
        self.class_weight = class_weight

    def __call__(self, y_true, y_pred):
        total = 0

        for label, proba in y_pred.items():
            if y_true == label:
                total += self.class_weight.get(label, 1.0) * math.log(clamp_proba(proba))

        return -total

    def gradient(self, y_true, y_pred):
        return {
            label: (
                self.class_weight.get(label, 1.0)
                * (clamp_proba(y_pred.get(label, 0.0)) - (y_true == label))
            )
            for label in {*y_pred.keys(), y_true}
        }


class Hinge(BinaryLoss):
    r"""Computes the hinge loss.

    Mathematically, it is defined as

    $$L = max(0, 1 - p_i * y_i)$$

    It's gradient w.r.t. to $p_i$ is

    $$
    \\frac{\\partial L}{\\partial y_i} = \\left\{
    \\begin{array}{ll}
        \\ 0  &   p_iy_i \geqslant 1  \\\\
        \\ - y_i & p_iy_i < 1
    \\end{array}
    \\right.
    $$

    Parameters
    ----------
    threshold
        Margin threshold. 1 yield the loss used in SVMs, whilst 0 is equivalent to the loss used in
        the Perceptron algorithm.

    Examples
    --------

    >>> from river import optim

    >>> loss = optim.losses.Hinge(threshold=1)
    >>> loss(1, .2)
    0.8

    >>> loss.gradient(1, .2)
    -1

    """

    def __init__(self, threshold=1.0):
        self.threshold = threshold

    def __call__(self, y_true, y_pred):
        y_true = y_true * 2 - 1  # [0, 1] -> [-1, 1]

        if isinstance(y_true, np.ndarray):
            return np.maximum(self.threshold - y_true * y_pred, 0)

        return max(self.threshold - y_true * y_pred, 0)

    def gradient(self, y_true, y_pred):
        y_true = y_true * 2 - 1  # [0, 1] -> [-1, 1]

        if isinstance(y_true, np.ndarray):
            return np.where(y_true * y_pred < self.threshold, -y_true, 0)

        if y_true * y_pred <= self.threshold:
            return -y_true
        return 0


class EpsilonInsensitiveHinge(RegressionLoss):
    """Epsilon-insensitive hinge loss.

    Parameters
    ----------
    eps

    """

    def __init__(self, eps=0.1):
        self.eps = eps

    def __call__(self, y_true, y_pred):
        y_true = y_true * 2 - 1  # [0, 1] -> [-1, 1]

        if isinstance(y_true, np.ndarray):
            return np.maximum(np.abs(y_pred - y_true) - self.eps, 0)

        return max(math.fabs(y_pred - y_true) - self.eps, 0)

    def gradient(self, y_true, y_pred):
        y_true = y_true * 2 - 1  # [0, 1] -> [-1, 1]

        if isinstance(y_true, np.ndarray):
            gradients = np.zeros_like(y_true)
            gradients[y_pred > y_true + self.eps] = 1
            gradients[y_pred + self.eps < y_true] = -1
            return gradients

        if y_pred > y_true + self.eps:
            return 1
        elif y_pred + self.eps < y_true:
            return -1
        return 0


class Log(BinaryLoss):
    """Logarithmic loss.

    This loss function expects each provided `y_pred` to be a logit. In other words if must be
    the raw output of a linear model or a neural network.

    Parameters
    ----------
    weight_pos
    weight_neg

    References
    ----------
    [^1]: [Logit Wikipedia page](https://www.wikiwand.com/en/Logit>)

    """

    def __init__(self, weight_pos=1.0, weight_neg=1.0):
        self.weight_pos = weight_pos
        self.weight_neg = weight_neg

    def __call__(self, y_true, y_pred):
        if isinstance(y_true, np.ndarray):
            weights = np.where(y_true == 0, self.weight_neg, self.weight_pos)
            y_true = 2 * y_true - 1  # map {0, 1} to {-1, 1}
            z = y_pred * y_true
            return weights * np.log(1.0 + np.exp(-z))

        weight = self.weight_pos
        if y_true == 0:
            y_true = -1
            weight = self.weight_neg
        else:
            y_true = int(y_true)

        z = y_pred * y_true
        if z > 18.0:
            return weight * math.exp(-z)
        if z < -18.0:
            return weight * -z
        return weight * math.log(1.0 + math.exp(-z))

    def gradient(self, y_true, y_pred):
        if isinstance(y_true, np.ndarray):
            weights = np.where(y_true == 0, self.weight_neg, self.weight_pos)
            y_true = 2 * y_true - 1  # map {0, 1} to {-1, 1}
            z = y_pred * y_true
            return weights * -y_true / (np.exp(z) + 1.0)

        weight = self.weight_pos
        if y_true == 0:
            y_true = -1
            weight = self.weight_neg
        else:
            y_true = int(y_true)

        z = y_pred * y_true
        if z > 18.0:
            return weight * math.exp(-z) * -y_true
        if z < -18.0:
            return weight * -y_true
        return weight * -y_true / (math.exp(z) + 1.0)


class Quantile(RegressionLoss):
    """Quantile loss.

    Parameters
    ----------
    alpha
        Desired quantile to attain.

    Examples
    --------

    >>> from river import optim

    >>> loss = optim.losses.Quantile(0.5)
    >>> loss(1, 3)
    1.0

    >>> loss.gradient(1, 3)
    0.5

    >>> loss.gradient(3, 1)
    -0.5

    References
    ----------
    [^1]: [Wikipedia article on quantile regression](https://www.wikiwand.com/en/Quantile_regression)
    [^2]: [Derivative from WolframAlpha](https://www.wolframalpha.com/input/?i=derivative+(y+-+p)+*+(alpha+-+Boole(y+-+p))+wrt+p)

    """

    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def __call__(self, y_true, y_pred):
        diff = y_pred - y_true
        return (self.alpha - (diff < 0)) * diff

    def gradient(self, y_true, y_pred):
        return (y_true < y_pred) - self.alpha


class Squared(RegressionLoss):
    """Squared loss, also known as the L2 loss.

    Mathematically, it is defined as

    $$L = (p_i - y_i) ^ 2$$

    It's gradient w.r.t. to $p_i$ is

    $$\\frac{\\partial L}{\\partial p_i} = 2 (p_i - y_i)$$

    One thing to note is that this convention is consistent with Vowpal Wabbit and PyTorch, but
    not with scikit-learn. Indeed, scikit-learn divides the loss by 2, making the 2 disappear in
    the gradient.

    Examples
    --------

    >>> from river import optim

    >>> loss = optim.losses.Squared()
    >>> loss(-4, 5)
    81
    >>> loss.gradient(-4, 5)
    18
    >>> loss.gradient(5, -4)
    -18

    """

    def __call__(self, y_true, y_pred):
        return (y_pred - y_true) * (y_pred - y_true)

    def gradient(self, y_true, y_pred):
        return 2 * (y_pred - y_true)


class Huber(RegressionLoss):
    """Huber loss.

    Variant of the squared loss that is robust to outliers.

    Parameters
    ----------
    epsilon

    References
    ----------
    1. [Huber loss function - Wikipedia](https://en.wikipedia.org/wiki/Huber_Loss_Function)

    """

    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon

    def __call__(self, y_true, y_pred):
        r = y_pred - y_true

        if isinstance(y_true, np.ndarray):
            abs_r = np.abs(r)
            return np.where(
                abs_r <= self.epsilon,
                0.5 * r * r,
                self.epsilon * abs_r - (0.5 * self.epsilon * self.epsilon),
            )

        abs_r = abs(r)
        if abs_r <= self.epsilon:
            return 0.5 * r * r
        return self.epsilon * abs_r - (0.5 * self.epsilon * self.epsilon)

    def gradient(self, y_true, y_pred):
        r = y_pred - y_true

        if isinstance(y_true, np.ndarray):
            abs_r = np.abs(r)
            return np.where(
                abs_r <= self.epsilon, r, np.where(r > 0.0, self.epsilon, -self.epsilon)
            )

        abs_r = abs(r)
        if abs_r <= self.epsilon:
            return r
        elif r > 0.0:
            return self.epsilon
        return -self.epsilon


class BinaryFocalLoss(BinaryLoss):
    """Binary focal loss.

    This implements the "star" algorithm from the appendix of the focal loss paper.

    Parameters
    ----------
    gamma
    beta

    References
    ----------
    1. [Lin, T.Y., Goyal, P., Girshick, R., He, K. and Dollár, P., 2017. Focal loss for dense object detection. In Proceedings of the IEEE international conference on computer vision (pp. 2980-2988)](https://arxiv.org/pdf/1708.02002.pdf)

    """

    def __init__(self, gamma=2, beta=1):
        self.gamma = gamma
        self.beta = beta

    def __call__(self, y_true, y_pred):
        y_true = y_true * 2 - 1  # [0, 1] -> [-1, 1]

        xt = y_true * y_pred

        if isinstance(y_true, np.ndarray):
            pt = 1.0 / (1 + np.exp(-(self.gamma * xt + self.beta)))
            return -np.log(pt) / self.gamma

        pt = utils.math.sigmoid(self.gamma * xt + self.beta)
        return -math.log(pt) / self.gamma

    def gradient(self, y_true, y_pred):
        y_true = y_true * 2 - 1  # [0, 1] -> [-1, 1]

        xt = y_true * y_pred

        if isinstance(y_true, np.ndarray):
            pt = 1.0 / (1 + np.exp(-(self.gamma * xt + self.beta)))
            return y_true * (pt - 1)

        pt = utils.math.sigmoid(self.gamma * xt + self.beta)
        return y_true * (pt - 1)


class Poisson(RegressionLoss):
    """Poisson loss.

    The Poisson loss is usually more suited for regression with count data than the squared loss.

    Mathematically, it is defined as

    $$L = exp(p_i) - y_i \\times p_i$$

    It's gradient w.r.t. to $p_i$ is

    $$\\frac{\\partial L}{\\partial p_i} = exp(p_i) - y_i$$

    """

    def __call__(self, y_true, y_pred):
        if isinstance(y_pred, np.ndarray):
            return np.exp(y_pred) - y_true * y_pred
        return math.exp(y_pred) - y_true * y_pred

    def gradient(self, y_true, y_pred):
        if isinstance(y_pred, np.ndarray):
            return np.exp(y_pred) - y_true
        return math.exp(y_pred) - y_true

    def mean_func(self, y_pred):
        if isinstance(y_pred, np.ndarray):
            return np.exp(y_pred)
        return math.exp(y_pred)
