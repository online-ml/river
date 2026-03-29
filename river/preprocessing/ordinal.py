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
    categories
        Categories (unique values) per feature:
            `None` : Determine categories automatically from the training data.

            dict of dicts : Expected categories for each feature. The outer dict maps each feature to its inner dict.
            The inner dict maps each category to its code.

        The used categories can be found in the `values` attribute.
    unknown_value
        The value to use for unknown categories seen during `transform_one`. Unknown categories
        will be mapped to an integer once they are seen during `learn_one`. This value can be set
        to `None` in order to categories to `None` if they've never been seen before.
    none_value
        The value to encode `None` with.

    Attributes
    ----------
    values
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

    Like in `scikit-learn`, you can also specify the expected categories manually.
    This is handy when you want to constrain category encoding space
    to e.g. top 20% most popular category values you've picked in advance.

    >>> categories = {'country': {'France': 1},
    ...               'place': {'Burger King': 2, 'Starbucks': 3}}
    >>> encoder = preprocessing.OrdinalEncoder(categories=categories)
    >>> for x in X:
    ...     print(encoder.transform_one(x))
    ...     encoder.learn_one(x)
    {'country': 1, 'place': 0}
    {'country': -1, 'place': -1}
    {'country': 0, 'place': 2}
    {'country': 1, 'place': 2}
    {'country': 0, 'place': 3}
    {'country': 0, 'place': 3}
    {'country': 0, 'place': 0}
    {'country': -1, 'place': -1}

    >>> import pandas as pd
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
        categories: dict | None = None,
        unknown_value: int | None = 0,
        none_value: int = -1,
    ):
        self.unknown_value = unknown_value
        self.none_value = none_value
        self.categories = categories
        self.values: collections.defaultdict | dict | None = None

        if self.categories is None:
            # We're going to store the categories in a dict of dicts. The outer dict will map each
            # feature to its inner dict. The inner dict will map each category to its code.
            self.values = collections.defaultdict(dict)
        else:
            self.values = self.categories

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
        code = len(self.values[feature])  # type: ignore[index]
        for reserved in self._reserved_category_codes:
            if reserved <= code:
                code += 1
        return code

    def _encode_value(self, feature: typing.Hashable, value):
        if value is None:
            return self.none_value
        return self.values[feature].get(value, self.unknown_value)  # type: ignore[index]

    def transform_one(self, x):
        return {i: self._encode_value(i, xi) for i, xi in x.items()}

    def learn_one(self, x):
        if self.categories is None:
            for i, xi in x.items():
                if xi is not None and xi not in self.values[i]:
                    self.values[i][xi] = self._next_category_code(i)

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
        if self.categories is None:
            for i in X.columns:
                for xi in X[i].dropna().unique():
                    if xi not in self.values[i]:
                        self.values[i][xi] = self._next_category_code(i)
