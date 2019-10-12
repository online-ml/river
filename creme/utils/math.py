import functools
import math
import operator

import numpy as np


__all__ = [
    'chain_dot',
    'clamp',
    'dot',
    'minkowski_distance',
    'norm',
    'prod',
    'sigmoid',
    'softmax'
]


def minkowski_distance(a, b, p):
    """Minkowski distance.

    Parameters:
        a (dict)
        b (dict)
        p (int): Parameter for the Minkowski distance. When ``p=1``, this is equivalent to using
            the Manhattan distance. When ``p=2``, this is equivalent to using the Euclidean
            distance.

    """
    return sum((abs(a.get(k, 0.) - b.get(k, 0.))) ** p for k in set([*a.keys(), *b.keys()]))


def softmax(y_pred):
    """Normalizes a dictionary of predicted probabilities, in-place."""

    if not y_pred:
        return y_pred

    maximum = max(y_pred.values())
    total = 0.

    for c, p in y_pred.items():
        y_pred[c] = math.exp(p - maximum)
        total += y_pred[c]

    for c in y_pred:
        y_pred[c] /= total

    return y_pred


def prod(iterable):
    return functools.reduce(operator.mul, iterable, 1)


def dot(x: dict, y: dict):
    """Returns the dot product of two vectors represented as dicts.

    Example:

        >>> x = {'x0': 1, 'x1': 2}
        >>> y = {'x1': 21, 'x2': 3}
        >>> dot(x, y)
        42.0

    """

    if len(x) < len(y):
        return sum(xi * y.get(i, 0.) for i, xi in x.items())
    return sum(x.get(i, 0.) * yi for i, yi in y.items())


def chain_dot(*xs):
    """Returns the dot product of multiple vectors represented as dicts.

    Example:

        >>> x = {'x0': 1, 'x1': 2, 'x2': 1}
        >>> y = {'x1': 21, 'x2': 3}
        >>> z = {'x1': 2, 'x2': 1 / 3}
        >>> chain_dot(x, y, z)
        85.0

    """
    keys = min(xs, key=len)
    return sum(prod(x.get(i, 0) for x in xs) for i in keys)


def sigmoid(x: float):
    if x < -30:
        return 0
    if x > 30:
        return 1
    return 1 / (1 + math.exp(-x))


def clamp(x: float, minimum=0., maximum=1.):
    return max(min(x, maximum), minimum)


def norm(x, order=None):
    return np.linalg.norm(list(x.values()), ord=order)
