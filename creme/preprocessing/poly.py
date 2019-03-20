import itertools
import functools
import operator

from .. import base


def prod(iterable):
    return functools.reduce(operator.mul, iterable, 1)


def powerset(iterable, min_size, max_size, with_replacement=False):
    """powerset([A, B, C], 1, 2) --> (A,) (B,) (C,) (A, B) (A, C) (B, C)"""
    combiner = itertools.combinations_with_replacement \
        if with_replacement \
        else itertools.combinations
    sizes = range(min_size, max_size + 1)
    return itertools.chain.from_iterable(combiner(list(iterable), size) for size in sizes)


class PolynomialExtender(base.Transformer):
    """

    Example:

    ::

        >>> from creme import preprocessing

        >>> X = [
        ...     {'x1': 0, 'x2': 1},
        ...     {'x1': 2, 'x2': 3},
        ...     {'x1': 4, 'x2': 5}
        ... ]

        >>> poly = preprocessing.PolynomialExtender(degree=2)
        >>> for x in X:
        ...     print(poly.fit_one(x).transform_one(x))
        {'x1': 0, 'x2': 1, 'x1*x1': 0, 'x1*x2': 0, 'x2*x2': 1}
        {'x1': 2, 'x2': 3, 'x1*x1': 4, 'x1*x2': 6, 'x2*x2': 9}
        {'x1': 4, 'x2': 5, 'x1*x1': 16, 'x1*x2': 20, 'x2*x2': 25}

        >>> poly = preprocessing.PolynomialExtender(degree=2, interaction_only=True)
        >>> for x in X:
        ...     print(poly.fit_one(x).transform_one(x))
        {'x1': 0, 'x2': 1, 'x1*x2': 0}
        {'x1': 2, 'x2': 3, 'x1*x2': 6}
        {'x1': 4, 'x2': 5, 'x1*x2': 20}

    """

    def __init__(self, degree=2, interaction_only=False):
        self.degree = degree
        self.interaction_only = interaction_only
        self.iterator = functools.partial(
            powerset,
            min_size=1,
            max_size=degree,
            with_replacement=not interaction_only
        )

    def transform_one(self, x):
        return {
            '*'.join(map(str, combo)): prod(x[c] for c in combo)
            for combo in self.iterator(x.keys())
        }
