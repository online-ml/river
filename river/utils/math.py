"""Mathematical utility functions (intended for internal purposes).

A lot of this is experimental and has a high probability of changing in the future.

"""
from __future__ import annotations

import functools
import itertools
import math
import operator

import numpy as np
import scipy as sp

__all__ = [
    "argmax",
    "chain_dot",
    "clamp",
    "dot",
    "dotvecmat",
    "matmul2d",
    "minkowski_distance",
    "norm",
    "outer",
    "prod",
    "sigmoid",
    "sign",
    "sherman_morrison",
    "softmax",
    "woodbury_matrix",
    "log_sum_2_exp",
]


def dotvecmat(x, A):
    """Vector times matrix from left side, i.e. transpose(x)A.

    Parameters
    ----------
    x
    A

    Examples
    --------

    >>> from river import utils

    >>> x = {0: 4, 1: 5}

    >>> A = {
    ...     (0, 0): 0, (0, 1): 1,
    ...     (1, 0): 2, (1, 1): 3
    ... }

    >>> C = utils.math.dotvecmat(x, A)
    >>> print(C)
    {0: 10.0, 1: 19.0}

    """

    C = {}

    for (i, xi), ((j, k), ai) in itertools.product(x.items(), A.items()):
        if i != j:
            continue

        C[k] = C.get(k, 0.0) + xi * ai

    return C


def matmul2d(A, B):
    """Multiplication for 2D matrices.

    Parameters
    ----------
    A
    B

    Examples
    --------

    >>> import pprint
    >>> from river import utils

    >>> A = {
    ...     (0, 0): 2, (0, 1): 0, (0, 2): 4,
    ...     (1, 0): 5, (1, 1): 6, (1, 2): 0
    ... }

    >>> B = {
    ...     (0, 0): 1, (0, 1): 1, (0, 2): 0, (0, 3): 0,
    ...     (1, 0): 2, (1, 1): 0, (1, 2): 1, (1, 3): 3,
    ...     (2, 0): 4, (2, 1): 0, (2, 2): 0, (2, 3): 0
    ... }

    >>> C = utils.math.matmul2d(A, B)
    >>> pprint.pprint(C)
    {(0, 0): 18.0,
        (0, 1): 2.0,
        (0, 2): 0.0,
        (0, 3): 0.0,
        (1, 0): 17.0,
        (1, 1): 5.0,
        (1, 2): 6.0,
        (1, 3): 18.0}

    """
    C = {}

    for ((i, k1), x), ((k2, j), y) in itertools.product(A.items(), B.items()):
        if k1 != k2:
            continue
        C[i, j] = C.get((i, j), 0.0) + x * y

    return C


def outer(u: dict, v: dict) -> dict:
    """Outer-product between two vectors.

    Parameters
    ----------
    u
    v

    Examples
    --------

    >>> import pprint
    >>> from river import utils

    >>> u = dict(enumerate((1, 2, 3)))
    >>> v = dict(enumerate((2, 4, 8)))

    >>> uTv = utils.math.outer(u, v)
    >>> pprint.pprint(uTv)
    {(0, 0): 2,
        (0, 1): 4,
        (0, 2): 8,
        (1, 0): 4,
        (1, 1): 8,
        (1, 2): 16,
        (2, 0): 6,
        (2, 1): 12,
        (2, 2): 24}

    """
    return {(ki, kj): vi * vj for (ki, vi), (kj, vj) in itertools.product(u.items(), v.items())}


def minkowski_distance(a: dict, b: dict, p: int):
    """Minkowski distance.

    Parameters
    ----------
    a
    b
    p
        Parameter for the Minkowski distance. When `p=1`, this is equivalent to using the
        Manhattan distance. When `p=2`, this is equivalent to using the Euclidean distance.

    """
    return sum((abs(a.get(k, 0.0) - b.get(k, 0.0))) ** p for k in {*a.keys(), *b.keys()}) ** (1 / p)


def softmax(y_pred: dict):
    """Normalizes a dictionary of predicted probabilities, in-place.

    Parameters
    ----------
    y_pred

    """

    if not y_pred:
        return y_pred

    maximum = max(y_pred.values())
    total = 0.0

    for c, p in y_pred.items():
        y_pred[c] = math.exp(p - maximum)
        total += y_pred[c]

    for c in y_pred:
        y_pred[c] /= total

    return y_pred


def prod(iterable):
    """Product function.

    Parameters
    ----------
    iterable

    """
    return functools.reduce(operator.mul, iterable, 1)


def dot(x: dict, y: dict):
    """Returns the dot product of two vectors represented as dicts.

    Parameters
    ----------
    x
    y

    Examples
    --------

    >>> from river import utils

    >>> x = {'x0': 1, 'x1': 2}
    >>> y = {'x1': 21, 'x2': 3}

    >>> utils.math.dot(x, y)
    42

    """

    if len(x) < len(y):
        return sum(xi * y[i] for i, xi in x.items() if i in y)
    return sum(x[i] * yi for i, yi in y.items() if i in x)


def chain_dot(*xs):
    """Returns the dot product of multiple vectors represented as dicts.

    Parameters
    ----------
    xs

    Examples
    --------

    >>> from river import utils

    >>> x = {'x0': 1, 'x1': 2, 'x2': 1}
    >>> y = {'x1': 21, 'x2': 3}
    >>> z = {'x1': 2, 'x2': 1 / 3}

    >>> utils.math.chain_dot(x, y, z)
    85.0

    """
    keys = min(xs, key=len)
    return sum(prod(x.get(i, 0) for x in xs) for i in keys)


def sigmoid(x: float):
    """Sigmoid function.

    Parameters
    ----------
    x

    """
    if x < -30:
        return 0
    if x > 30:
        return 1
    return 1 / (1 + math.exp(-x))


def clamp(x: float, minimum=0.0, maximum=1.0):
    """Clamp a number.

    This is a synonym of clipping.

    Parameters
    ----------
    x
    minimum
    maximum

    """
    return max(min(x, maximum), minimum)


def norm(x: dict, order=None):
    """Compute the norm of a dictionaries values.

    Parameters
    ----------
    x
    order

    """
    return np.linalg.norm(list(x.values()), ord=order)


def sign(x: float):
    """Sign function.

    Parameters
    ----------
    x

    """
    return -1 if x < 0 else (1 if x > 0 else 0)


def argmax(lst: list):
    """Argmax function.

    Parameters
    ----------
    lst

    """
    return max(range(len(lst)), key=lst.__getitem__)


def sherman_morrison(A: np.ndarray, u: np.ndarray, v: np.ndarray):
    """Sherman-Morrison formula.

    This is an inplace function.

    Parameters
    ----------
    A
    u
    v

    References
    ----------
    [^1]: [Fast rank-one updates to matrix inverse? — Tim Vieira](https://timvieira.github.io/blog/post/2021/03/25/fast-rank-one-updates-to-matrix-inverse/)

    """
    Au = A @ u
    alpha = -1 / (1 + v.T @ Au)
    sp.linalg.blas.dger(alpha, Au, v.T @ A, a=A, overwrite_a=1)


def woodbury_matrix(A: np.ndarray, U: np.ndarray, V: np.ndarray):
    """Woodbury matrix identity.

    This is an inplace function.

    Parameters
    ----------
    A
    U
    V

    References
    ----------
    [^1]: [Matrix inverse mini-batch updates — Max Halford](https://maxhalford.github.io/blog/matrix-inverse-mini-batch/)

    """
    eye = np.eye(len(V))
    Au = A @ U
    A -= Au @ np.linalg.inv(eye + V @ Au) @ V @ A


def log_sum_2_exp(a: float, b: float) -> float:
    """Computation of log( (e^a + e^b) / 2) in an overflow-proof way

    Parameters
    ----------
    a
        First number

    b
        Second number
    """
    # TODO: if |a - b| > 50 skip
    # TODO: try several log and exp implementations
    if a > b:
        return a + math.log((1 + math.exp(b - a)) / 2)
    else:
        return b + math.log((1 + math.exp(a - b)) / 2)
