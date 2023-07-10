from __future__ import annotations

import collections
import functools
import itertools

import numpy as np
import pandas as pd

from river import base


def make_counter(skip):
    return (i for i in itertools.count() if i not in skip)


class OrdinalEncoder(base.MiniBatchTransformer):
    """Ordinal encoder.

    This transformer maps each feature to integers. It can useful when a feature has string values
    (i.e. categorical variables).

    Parameters
    ----------
    unknown_value
        The value to use for unknown categories seen during `transform_one`. Unknown categories
        will be mapped to an integer once they are seen during `learn_one`. This value can be set
        to `None` in order to categories to `None` if they've never been seen before.
    none_value
        The value to encode `None` with.

    Attributes
    ----------
    categories
        A dict of dicts. The outer dict maps each feature to its inner dict. The inner dict maps
        each category to its code.

    Examples
    --------

    >>> from river import preprocessing

    >>> X = [
    ...     {"country": "France", "place": "Taco Bell"},
    ...     {"country": None, "place": None},
    ...     {"country": "Sweden", "place": "Burger King"},
    ...     {"country": "France", "place": "Burger King"},
    ...     {"country": "Russia", "place": "Starbucks"},
    ...     {"country": "Russia", "place": "Starbucks"},
    ...     {"country": "Sweden", "place": "Taco Bell"},
    ...     {"country": None, "place": None},
    ... ]

    >>> encoder = preprocessing.OrdinalEncoder()
    >>> for x in X:
    ...     print(encoder.transform_one(x))
    ...     encoder = encoder.learn_one(x)
    {'country': 0, 'place': 0}
    {'country': -1, 'place': -1}
    {'country': 0, 'place': 0}
    {'country': 1, 'place': 2}
    {'country': 0, 'place': 0}
    {'country': 3, 'place': 3}
    {'country': 2, 'place': 1}
    {'country': -1, 'place': -1}

    >>> xb1 = pd.DataFrame(X[0:4], index=[0, 1, 2, 3])
    >>> xb2 = pd.DataFrame(X[4:8], index=[4, 5, 6, 7])

    >>> encoder = preprocessing.OrdinalEncoder()
    >>> encoder.transform_many(xb1)
       country  place
    0        0      0
    1       -1     -1
    2        0      0
    3        0      0

    >>> encoder = encoder.learn_many(xb1)
    >>> encoder.transform_many(xb2)
       country  place
    4        0      0
    5        0      0
    6        2      1
    7       -1     -1

    """

    def __init__(
        self,
        unknown_value: int | None = 0,
        none_value: int = -1,
    ):
        self.unknown_value = unknown_value
        self.none_value = none_value

        # We're going to have one auto-incrementing counter per feature. This counter will generate
        # the category codes for each feature.
        self._counters: collections.defaultdict = collections.defaultdict(
            functools.partial(make_counter, {unknown_value, none_value})
        )

        # We're going to store the categories in a dict of dicts. The outer dict will map each
        # feature to its inner dict. The inner dict will map each category to its code.
        self.categories: collections.defaultdict = collections.defaultdict(dict)

    def transform_one(self, x):
        return {
            i: self.none_value if xi is None else self.categories[i].get(xi, self.unknown_value)
            for i, xi in x.items()
        }

    def learn_one(self, x):
        for i, xi in x.items():
            if xi is not None and xi not in self.categories[i]:
                self.categories[i][xi] = next(self._counters[i])
        return self

    def transform_many(self, X):
        return pd.DataFrame(
            {
                i: pd.Series(
                    X[i]
                    .map({**self.categories[i], None: self.none_value})
                    .fillna(self.unknown_value),
                    dtype=np.int64,
                )
                for i in X.columns
            }
        )

    def learn_many(self, X, y=None):
        for i in X.columns:
            for xi in X[i].dropna().unique():
                if xi not in self.categories[i]:
                    self.categories[i][xi] = next(self._counters[i])
        return self
