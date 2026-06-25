from __future__ import annotations

import collections
import typing

import narwhals.stable.v2 as nw

from river import base, utils

if typing.TYPE_CHECKING:
    from narwhals.stable.v2.typing import IntoDataFrame, IntoDataFrameT, IntoSeries


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
        categories: dict[base.typing.FeatureName, dict[typing.Hashable, int]] | None = None,
        unknown_value: int | None = 0,
        none_value: int = -1,
    ):
        self.unknown_value = unknown_value
        self.none_value = none_value
        self.categories = categories
        self.values: (
            collections.defaultdict[base.typing.FeatureName, dict[typing.Hashable, int]]
            | dict[base.typing.FeatureName, dict[typing.Hashable, int]]
        )

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
        code = len(self.values[feature])
        for reserved in self._reserved_category_codes:
            if reserved <= code:
                code += 1
        return code

    def _encode_value(self, feature: typing.Hashable, value):
        if value is None:
            return self.none_value
        return self.values[feature].get(value, self.unknown_value)

    def transform_one(self, x):
        return {i: self._encode_value(i, xi) for i, xi in x.items()}

    def learn_one(self, x):
        if self.categories is None:
            for i, xi in x.items():
                if xi is not None and xi not in self.values[i]:
                    self.values[i][xi] = self._next_category_code(i)

    def transform_many(self, X: IntoDataFrameT) -> IntoDataFrameT:
        """Encode a mini-batch of features into integer codes.

        Missing cells (``None``/``NaN``/``pd.NA``) are mapped to ``none_value`` consistently
        across backends; values absent from the learned categories map to ``unknown_value``.

        Parameters
        ----------
        X
            A dataframe where each column is a categorical feature.

        """
        X_nw = utils.dataframe.into_frame(X)
        schema = X_nw.schema
        exprs: list[nw.Expr] = []
        for col, dtype in schema.items():
            mapping = self.values[col]
            # `replace_strict` maps each known category to its code and everything else to
            # `unknown_value` in one vectorised pass. An empty mapping (a column never seen during
            # `learn_many`) has no categories to match, so the whole column collapses to
            # `unknown_value`.
            if mapping:
                encoded = nw.col(col).replace_strict(
                    list(mapping.keys()),
                    list(mapping.values()),
                    default=self.unknown_value,
                    return_dtype=nw.Int64,
                )
            else:
                encoded = nw.lit(self.unknown_value, dtype=nw.Int64)
            # Missing cells map to `none_value` separately, since `replace_strict` leaves them
            # untouched. A numeric `NaN` is missing too, but polars/pyarrow keep it distinct from
            # null, so it is folded in explicitly (mirroring `learn_many`, which also skips it).
            # `is_nan` errors on non-numeric dtypes, hence the guard.
            is_missing = nw.col(col).is_null()
            if dtype.is_numeric():
                is_missing = is_missing | nw.col(col).is_nan()

            exprs.append(
                nw.when(is_missing)  # type:ignore[arg-type]
                .then(nw.lit(self.none_value, dtype=nw.Int64))
                .otherwise(encoded)
                .alias(col)
            )
        return X_nw.select(exprs).to_native()

    def learn_many(self, X: IntoDataFrame, y: IntoSeries | None = None) -> None:
        if self.categories is None:
            X_nw = utils.dataframe.into_frame(X)
            schema = X_nw.schema
            # Skip missing cells before registering categories: numeric columns can carry NaN
            # (not caught by `is_null` on every backend), so exclude both for numeric dtypes.
            finite_masks = X_nw.select(
                [
                    ~(nw.col(name).is_nan() | nw.col(name).is_null())
                    if dtype.is_numeric()
                    else ~nw.col(name).is_null()
                    for name, dtype in schema.items()
                ]
            )
            for col_name in schema.names():
                # `maintain_order=True` assigns codes in first-appearance order on every backend,
                # matching `learn_one` and keeping the encoding reproducible across backends
                # (polars' default `unique` does not preserve order).
                for value in (
                    X_nw[col_name].filter(finite_masks[col_name]).unique(maintain_order=True)
                ):
                    if value not in self.values[col_name]:
                        self.values[col_name][value] = self._next_category_code(col_name)
