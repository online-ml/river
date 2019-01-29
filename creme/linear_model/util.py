import math


__all__ = ['dot', 'sigmoid']


def dot(x: dict, y: dict):
    """Returns the dot product of two vectors represented as dicts.

    Example:

        >>> x = {'x0': 1, 'x1': 2}
        >>> y = {'x1': 21, 'x2': 3}
        >>> dot(x, y)
        42

    """
    return sum(xi * y.get(i, 0) for i, xi in min(x, y, key=len).items())


def sigmoid(x: float):
    return 1 / (1 + math.exp(-x))
