import itertools
import functools

from .. import base
from .. import utils


__all__ = ['PolynomialExtender']


def powerset(iterable, min_size, max_size, with_replacement=False):
    """powerset([A, B, C], 1, 2) --> (A,) (B,) (C,) (A, B) (A, C) (B, C)"""
    combiner = itertools.combinations_with_replacement \
        if with_replacement \
        else itertools.combinations
    sizes = range(min_size, max_size + 1)
    return itertools.chain.from_iterable(combiner(list(iterable), size) for size in sizes)


class PolynomialExtender(base.Transformer):
    """Polynomial feature extender.

    Generate features consisting of all polynomial combinations of the features with degree less
    than or equal to the specified degree.
    Be aware that the number of outputed features scales polynomially in the number of input
    features and exponentially in the degree. High degrees can cause overfitting.

    Parameters:
        degree (int): The maximum degree of the polynomial features.
        interaction_only (bool): If ``True`` then only combinations that include an element at most
            once will be computed.
        include_bias (bool): Whether or not to include a dummy feature which is always equal to 1.

    Example:

        ::

            >>> from creme import preprocessing

            >>> X = [
            ...     {'x1': 0, 'x2': 1},
            ...     {'x1': 2, 'x2': 3},
            ...     {'x1': 4, 'x2': 5}
            ... ]

            >>> poly = preprocessing.PolynomialExtender(degree=2, include_bias=True)
            >>> for x in X:
            ...     print(poly.fit_one(x).transform_one(x))
            {'x1': 0, 'x2': 1, 'x1*x1': 0, 'x1*x2': 0, 'x2*x2': 1, 'bias': 1}
            {'x1': 2, 'x2': 3, 'x1*x1': 4, 'x1*x2': 6, 'x2*x2': 9, 'bias': 1}
            {'x1': 4, 'x2': 5, 'x1*x1': 16, 'x1*x2': 20, 'x2*x2': 25, 'bias': 1}

            >>> X = [
            ...     {'x1': 0, 'x2': 1, 'x3': 2},
            ...     {'x1': 2, 'x2': 3, 'x3': 2},
            ...     {'x1': 4, 'x2': 5, 'x3': 2}
            ... ]

            >>> poly = preprocessing.PolynomialExtender(degree=3, interaction_only=True)
            >>> for x in X:
            ...     print(poly.fit_one(x).transform_one(x))
            {'x1': 0, 'x2': 1, 'x3': 2, 'x1*x2': 0, 'x1*x3': 0, 'x2*x3': 2, 'x1*x2*x3': 0}
            {'x1': 2, 'x2': 3, 'x3': 2, 'x1*x2': 6, 'x1*x3': 4, 'x2*x3': 6, 'x1*x2*x3': 12}
            {'x1': 4, 'x2': 5, 'x3': 2, 'x1*x2': 20, 'x1*x3': 8, 'x2*x3': 10, 'x1*x2*x3': 40}

    """

    def __init__(self, degree=2, interaction_only=False, include_bias=False, bias_name='bias'):
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.bias_name = bias_name
        self.enumerate = functools.partial(
            powerset,
            min_size=1,
            max_size=degree,
            with_replacement=not interaction_only
        )

    def transform_one(self, x):
        features = {
            '*'.join(map(str, combo)): utils.prod(x[c] for c in combo)
            for combo in self.enumerate(x.keys())
        }
        if self.include_bias:
            features[self.bias_name] = 1
        return features
