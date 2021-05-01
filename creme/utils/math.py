"""Mathematical utility functions (intended for internal purposes).

A lot of this is experimental and has a high probability of changing in the future.

"""
import functools
import itertools
import math
import operator

import numpy as np


__all__ = [
    'chain_dot',
    'clamp',
    'dot',
    'dotvecmat',
    'matmul2d',
    'minkowski_distance',
    'norm',
    'outer',
    'prod',
    'sigmoid',
    'sign',
    'sherman_morrison',
    'softmax',
    'woodbury_identity',
    'eye_like'
]


def sherman_morrison(A_inv: dict, u: dict, v: dict) -> dict:
    """Sherman–Morrison formula.

    This modifies `A_inv` inplace.

    Parameters:
        A_inv
        u
        v

    References:
        1. [Wikipedia article on the Sherman-Morrison formula](https://www.wikiwand.com/en/Sherman%E2%80%93Morrison_formula)s

    Example:

        >>> import pprint
        >>> from creme import utils

        >>> A_inv = {
        ...     (0, 0): 0.2,
        ...     (1, 1): 1,
        ...     (2, 2): 1
        ... }
        >>> u = {0: 1, 1: 2, 2: 3}
        >>> v = {0: 4}

        >>> inv = sherman_morrison(A_inv, u, v)
        >>> pprint.pprint(inv)
        {(0, 0): 0.111111,
            (1, 0): -0.888888,
            (1, 1): 1,
            (2, 0): -1.333333,
            (2, 2): 1}

    """

    den = 1 + dot(dotvecmat(u, A_inv), v)

    for k, v in matmul2d(matmul2d(A_inv, outer(u, v)), A_inv).items():
        A_inv[k] = A_inv.get(k, 0) - v / den

    return A_inv


def woodbury_identity(A_inv, U, V):
    """Woodbury identity.

    This modifies ``A_inv`` inplace.

    References:
        1. `Wikipedia article <https://www.wikiwand.com/en/Sherman%E2%80%93Morrison_formula>`_
    """
    
    DEN = inverse(diff_mat(eye_like(V), matmul2d(matmul2d(transpose(V), A_inv), U)))

    for k, S in matmul2d(matmul2d(matmul2d(matmul2d(A_inv, U), DEN), transpose(V)), A_inv).items():
        A_inv[k] = A_inv.get(k, 0) + S

    return A_inv

    
def eye_like(A):
    identity = {}
    
    for (i, j) in A:
        if i == j:
            identity[i ,j] = 1.
        else:
            identity[i, j] = 0.
    
    return identity


def inverse(A):
    """Inverse calculation of matrix.
    
    """
    k1, k2 = [sorted(set([i for i, j  in set(A)])), sorted(set([j for i, j in set(A)]))]
    A_inv_np = np.linalg.inv(np.array([[A[i, j] for i in k1] for j in k2]))
    A_inv = {}
    
    for i, u in enumerate(k1):
        for j, v in enumerate(k2):
            A_inv[u, v] = A_inv_np[i, j]
    
    return A_inv


def transpose(A):
    A_trans = {}
    
    for ((i, j), x) in A.items():
            A_trans[j, i] = x
    
    return A_trans


def sum_mat(A, B):

    return {
        k: A.get(k, 0.) + B.get(k, 0.) for k in set(A).union(B)
    }


def diff_mat(A, B):

    return {
        k: A.get(k, 0.) - B.get(k, 0.) for k in set(A).union(B)
    }       


def dotvecmat(x, A):

    C = {}

    for (i, xi), ((j, k), ai) in itertools.product(x.items(), A.items()):

        if i != j:
            continue

        C[k] = C.get(j, 0.) + xi * ai

    return C


def matmul2d(A, B):
    """Multiplication for 2D matrices.

    Parameters:
        A
        B

    Example:

        >>> import pprint
        >>> from creme import utils

        >>> A = {
        ...     (0, 0): 2, (0, 1): 0, (0, 2): 4,
        ...     (1, 0): 5, (1, 1): 6, (1, 2): 0
        ... }

        >>> B = {
        ...     (0, 0): 1, (0, 1): 1, (0, 2): 0, (0, 3): 0,
        ...     (1, 0): 2, (1, 1): 0, (1, 2): 1, (1, 3): 3,
        ...     (2, 0): 4, (2, 1): 0, (2, 2): 0, (2, 3): 0
        ... }

        >>> C = matmul2d(A, B)
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
        C[i, j] = C.get((i, j), 0.) + x * y

    return C


def outer(u: dict, v: dict) -> dict:
    """Outer-product between two vectors.

    Parameters:
        u
        v

    Example:

        >>> import pprint
        >>> from creme import utils

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
    return {
        (ki, kj): vi * vj
        for (ki, vi), (kj, vj) in itertools.product(u.items(), v.items())
    }


def minkowski_distance(a: dict, b: dict, p: int):
    """Minkowski distance.

    Parameters:
        a
        b
        p: Parameter for the Minkowski distance. When `p=1`, this is equivalent to using the
            Manhattan distance. When `p=2`, this is equivalent to using the Euclidean distance.

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

        >>> from creme import utils

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

    Example:

        ::

            >>> from creme import utils

            >>> x = {'x0': 1, 'x1': 2, 'x2': 1}
            >>> y = {'x1': 21, 'x2': 3}
            >>> z = {'x1': 2, 'x2': 1 / 3}

            >>> utils.math.chain_dot(x, y, z)
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


def sign(x: float):
    return -1 if x < 0 else (1 if x > 0 else 0)
