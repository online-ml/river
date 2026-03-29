from __future__ import annotations

import collections
import typing

import numpy as np
import pandas as pd

from river import base


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
    ...     encoder.learn_one(x)
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

    >>> encoder.learn_many(xb1)
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

        # We're going to store the categories in a dict of dicts. The outer dict will map each
        # feature to its inner dict. The inner dict will map each category to its code.
        self.categories: collections.defaultdict = collections.defaultdict(dict)

        # Codes reserved for unknown and missing values should never be used for categories.
        self._reserved_category_codes = tuple(
            sorted(
                {
                    value
                    for value in (self.unknown_value, self.none_value)
                    if isinstance(value, int) and value >= 0
                }
            )
        )

    def _next_category_code(self, feature: typing.Hashable) -> int:
        """Return the next available integer code for a feature.

        The code is derived from the number of categories already assigned, skipping over any
        reserved values (``unknown_value`` and ``none_value``).

        """
        code = len(self.categories[feature])
        for reserved in self._reserved_category_codes:
            if reserved <= code:
                code += 1
        return code

    def _encode_value(self, feature: typing.Hashable, value):
        if value is None:
            return self.none_value
        return self.categories[feature].get(value, self.unknown_value)

    def transform_one(self, x):
        return {i: self._encode_value(i, xi) for i, xi in x.items()}

    def learn_one(self, x):
        for i, xi in x.items():
            if xi is not None and xi not in self.categories[i]:
                self.categories[i][xi] = self._next_category_code(i)

    def transform_many(self, X):
        return pd.DataFrame(
            {
                i: pd.Series(
                    (self._encode_value(i, value) for value in X[i]),
                    index=X.index,
                    dtype=np.int64,
                )
                for i in X.columns
            }
        )

    def learn_many(self, X, y=None):
        for i in X.columns:
            for xi in X[i].dropna().unique():
                if xi not in self.categories[i]:
                    self.categories[i][xi] = self._next_category_code(i)
