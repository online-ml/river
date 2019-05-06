import functools
import math
import operator

import numpy as np


__all__ = [
    'chain_dot',
    'clip',
    'dot',
    'norm',
    'prod',
    'sigmoid',
    'softmax'
]


def softmax(y_pred):
    """Normalizes a dictionary of predicted probabilities, in-place."""
    exp = {c: math.exp(p) for c, p in y_pred.items()}
    total = sum(exp.values())
    return {c: exp[c] / total for c in y_pred}


def prod(iterable):
    return functools.reduce(operator.mul, iterable, 1)


def dot(x: dict, y: dict):
    """Returns the dot product of two vectors represented as dicts.

    Example:

        >>> x = {'x0': 1, 'x1': 2}
        >>> y = {'x1': 21, 'x2': 3}
        >>> dot(x, y)
        42

    """
    return sum(xi * y.get(i, 0) for i, xi in min(x, y, key=len).items())


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


def clip(x: float, minimum=0., maximum=1.):
    return max(min(x, maximum), minimum)


def norm(x, order=None):
    return np.linalg.norm(list(x.values()), ord=order)
