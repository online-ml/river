from __future__ import annotations

import collections
import typing

import narwhals.stable.v2 as nw

from river import base, utils

if typing.TYPE_CHECKING:
    import pandas as pd
    from narwhals.stable.v2.typing import IntoDataFrame, IntoDataFrameT

__all__ = ["OneHotEncoder"]


class OneHotEncoder(base.MiniBatchTransformer):
    """One-hot encoding.

    This transformer will encode every feature it is provided with.
    If a list or set is provided, this transformer will encode every entry in the list/set.
    You can apply it to a subset of features by composing it
     with `compose.Select` or `compose.SelectType`.

    Parameters
    ----------
    categories
        Categories (unique values) per feature:
            `None` : Determine categories automatically from the training data.

            dict of dicts : Expected categories for each feature. The outer dict maps each feature to its inner dict.
            The inner dict maps each category to its code.

        The used categories can be found in the `values` attribute.
    drop_zeros
        Whether or not 0s should be made explicit or not.
    drop_first
        Whether to get `k - 1` dummies out of `k` categorical levels by removing the first key.
        This is useful in some statistical models where perfectly collinear features cause
        problems.

    Attributes
    ----------
    values
        A dict of dicts. The outer dict maps each feature to its inner dict. The inner dict maps
        each category to its code.

    Examples
    --------

    Let us first create an example dataset.

    >>> from pprint import pprint
    >>> import random
    >>> import string

    >>> random.seed(42)
    >>> alphabet = list(string.ascii_lowercase)
    >>> X = [
    ...     {
    ...         'c1': random.choice(alphabet),
    ...         'c2': random.choice(alphabet),
    ...     }
    ...     for _ in range(4)
    ... ]
    >>> pprint(X)
    [{'c1': 'u', 'c2': 'd'},
     {'c1': 'a', 'c2': 'x'},
     {'c1': 'i', 'c2': 'h'},
     {'c1': 'h', 'c2': 'e'}]

    We can now apply one-hot encoding. All the provided are one-hot encoded, there is therefore
    no need to specify which features to encode.

    >>> from river import preprocessing

    >>> oh = preprocessing.OneHotEncoder()
    >>> for x in X[:2]:
    ...     oh.learn_one(x)
    ...     pprint(oh.transform_one(x))
    {'c1_u': 1, 'c2_d': 1}
    {'c1_a': 1, 'c1_u': 0, 'c2_d': 0, 'c2_x': 1}

    The `drop_zeros` parameter can be set to `True` if you don't want the past features to be included
    in the output. Otherwise, all the past features will be included in the output.

    >>> oh = preprocessing.OneHotEncoder(drop_zeros=True)
    >>> for x in X:
    ...     oh.learn_one(x)
    ...     pprint(oh.transform_one(x))
    {'c1_u': 1, 'c2_d': 1}
    {'c1_a': 1, 'c2_x': 1}
    {'c1_i': 1, 'c2_h': 1}
    {'c1_h': 1, 'c2_e': 1}

    You can encode only `k - 1` features out of `k` by setting `drop_first` to `True`.

    >>> oh = preprocessing.OneHotEncoder(drop_first=True, drop_zeros=True)
    >>> for x in X:
    ...     oh.learn_one(x)
    ...     pprint(oh.transform_one(x))
    {'c2_d': 1}
    {'c2_x': 1}
    {'c2_h': 1}
    {'c2_e': 1}

    Like in `scikit-learn`, you can also specify the expected categories manually.
    This is handy when you want to constrain category encoding space
    to e.g. top 20% most popular category values you've picked in advance.

    >>> categories = {'c1': {'a', 'h'}, 'c2': {'x', 'e'}}
    >>> oh = preprocessing.OneHotEncoder(categories=categories)
    >>> for x in X:
    ...     oh.learn_one(x)
    ...     pprint(oh.transform_one(x))
    {'c1_a': 0, 'c1_h': 0, 'c2_e': 0, 'c2_x': 0}
    {'c1_a': 1, 'c1_h': 0, 'c2_e': 0, 'c2_x': 1}
    {'c1_a': 0, 'c1_h': 0, 'c2_e': 0, 'c2_x': 0}
    {'c1_a': 0, 'c1_h': 1, 'c2_e': 1, 'c2_x': 0}

    >>> for key in sorted(oh.values.keys()):
    ...     print(key)
    ...     print(sorted(oh.values[key]))
    c1
    ['a', 'h']
    c2
    ['e', 'x']

    A subset of the features can be one-hot encoded by piping a `compose.Select` into the
    `OneHotEncoder`.

    >>> from river import compose

    >>> pp = compose.Select('c1') | preprocessing.OneHotEncoder()

    >>> for x in X:
    ...     pp.learn_one(x)
    ...     pprint(pp.transform_one(x))
    {'c1_u': 1}
    {'c1_a': 1, 'c1_u': 0}
    {'c1_a': 0, 'c1_i': 1, 'c1_u': 0}
    {'c1_a': 0, 'c1_h': 1, 'c1_i': 0, 'c1_u': 0}

    You can preserve the `c2` feature by using a union:

    >>> pp = compose.Select('c1') | preprocessing.OneHotEncoder()
    >>> pp += compose.Select('c2')

    >>> for x in X:
    ...     pp.learn_one(x)
    ...     pprint(pp.transform_one(x))
    {'c1_u': 1, 'c2': 'd'}
    {'c1_a': 1, 'c1_u': 0, 'c2': 'x'}
    {'c1_a': 0, 'c1_i': 1, 'c1_u': 0, 'c2': 'h'}
    {'c1_a': 0, 'c1_h': 1, 'c1_i': 0, 'c1_u': 0, 'c2': 'e'}

    Similar to the above examples, we can also pass values as a list. This will one-hot
    encode all of the entries individually.

    >>> X = [{'c1': ['u', 'a'], 'c2': ['d']},
    ...     {'c1': ['a', 'b'], 'c2': ['x']},
    ...     {'c1': ['i'], 'c2': ['h', 'z']},
    ...     {'c1': ['h', 'b'], 'c2': ['e']}]

    >>> oh = preprocessing.OneHotEncoder(drop_zeros=True)
    >>> for x in X:
    ...     oh.learn_one(x)
    ...     pprint(oh.transform_one(x))
    {'c1_a': 1, 'c1_u': 1, 'c2_d': 1}
    {'c1_a': 1, 'c1_b': 1, 'c2_x': 1}
    {'c1_i': 1, 'c2_h': 1, 'c2_z': 1}
    {'c1_b': 1, 'c1_h': 1, 'c2_e': 1}

    Processing mini-batches is also possible.

    >>> from pprint import pprint
    >>> import random
    >>> import string
    >>> import pandas as pd

    >>> random.seed(42)
    >>> alphabet = list(string.ascii_lowercase)
    >>> X = pd.DataFrame(
    ...     {
    ...         'c1': random.choice(alphabet),
    ...         'c2': random.choice(alphabet),
    ...     }
    ...     for _ in range(3)
    ... )
    >>> X
      c1 c2
    0  u  d
    1  a  x
    2  i  h

    >>> oh = preprocessing.OneHotEncoder(drop_zeros=True)
    >>> df = oh.transform_many(X)
    >>> df.sort_index(axis="columns")
       c1_a  c1_i  c1_u  c2_d  c2_h  c2_x
    0     0     0     1     1     0     0
    1     1     0     0     0     0     1
    2     0     1     0     0     1     0

    >>> oh = preprocessing.OneHotEncoder(drop_zeros=True, drop_first=True)
    >>> df = oh.transform_many(X)
    >>> df.sort_index(axis="columns")
       c1_i  c1_u  c2_d  c2_h  c2_x
    0     0     1     1     0     0
    1     0     0     0     0     1
    2     1     0     0     1     0

    Here's an example where the zeros are kept:

    >>> oh = preprocessing.OneHotEncoder(drop_zeros=False)
    >>> X_init = pd.DataFrame([{"c1": "Oranges", "c2": "Apples"}])
    >>> oh.learn_many(X_init)
    >>> oh.learn_many(X)

    >>> df = oh.transform_many(X)
    >>> df.sort_index(axis="columns")
       c1_Oranges  c1_a  c1_i  c1_u  c2_Apples  c2_d  c2_h  c2_x
    0           0     0     0     1          0     1     0     0
    1           0     1     0     0          0     0     0     1
    2           0     0     1     0          0     0     1     0

    >>> df.dtypes.sort_index()
    c1_Oranges    Sparse[uint8, 0]
    c1_a          Sparse[uint8, 0]
    c1_i          Sparse[uint8, 0]
    c1_u          Sparse[uint8, 0]
    c2_Apples     Sparse[uint8, 0]
    c2_d          Sparse[uint8, 0]
    c2_h          Sparse[uint8, 0]
    c2_x          Sparse[uint8, 0]
    dtype: object

    Explicit categories:

    >>> oh = preprocessing.OneHotEncoder(categories=categories)


    >>> oh.learn_many(X)
    >>> df = oh.transform_many(X)
    >>> df.sort_index(axis="columns")
       c1_a  c1_h  c2_e  c2_x
    0     0     0     0     0
    1     1     0     0     1
    2     0     0     0     0

    """

    def __init__(self, categories: dict | None = None, drop_zeros=False, drop_first=False):
        self.drop_zeros = drop_zeros
        self.drop_first = drop_first
        self.categories = categories
        self.values: collections.defaultdict | dict
        self._zero_dict: dict = {}

        if self.categories is None:
            self.values = collections.defaultdict(set)
        else:
            self.values = self.categories
            if not self.drop_zeros:
                self._zero_dict = {f"{i}_{v}": 0 for i, vals in self.values.items() for v in vals}

    def learn_one(self, x):
        if self.drop_zeros:
            return

        # NOTE: assume if category mappings are explicitly provided,
        # they're intended to be kept fixed.
        if self.categories is None:
            values = self.values
            zero_dict = self._zero_dict
            for i, xi in x.items():
                vi = values[i]
                if isinstance(xi, (list, set)):
                    for xj in xi:
                        if xj not in vi:
                            vi.add(xj)
                            zero_dict[f"{i}_{xj}"] = 0
                elif xi not in vi:
                    vi.add(xi)
                    zero_dict[f"{i}_{xi}"] = 0

    def transform_one(self, x, y=None):
        oh = {} if self.drop_zeros else self._zero_dict.copy()

        # Add 1
        # NOTE: assume if category mappings are explicitly provided,
        # no other category values are allowed for output. Aligns with `sklearn` behavior.
        if self.categories is None:
            for i, xi in x.items():
                if isinstance(xi, (list, set)):
                    for xj in xi:
                        oh[f"{i}_{xj}"] = 1
                else:
                    oh[f"{i}_{xi}"] = 1
        else:
            values = self.values
            for i, xi in x.items():
                vi = values[i]
                if isinstance(xi, (list, set)):
                    for xj in xi:
                        if xj in vi:
                            oh[f"{i}_{xj}"] = 1
                elif xi in vi:
                    oh[f"{i}_{xi}"] = 1

        if self.drop_first:
            oh.pop(min(oh))

        return oh

    def learn_many(self, X: IntoDataFrame) -> None:
        if self.drop_zeros:
            return

        # NOTE: assume if category mappings are explicitly provided,
        # they're intended to be kept fixed.
        if self.categories is None:
            values = self.values
            zero_dict = self._zero_dict
            X_nw = utils.dataframe.into_frame(X)
            for col in X_nw.columns:
                vi = values[col]
                for v in X_nw[col].unique():
                    if v not in vi:
                        vi.add(v)
                        zero_dict[f"{col}_{v}"] = 0

    def transform_many(self, X: IntoDataFrameT) -> IntoDataFrameT:
        """One-hot encode a mini-batch of features.

        Pandas keeps the historical fast path returning ``Sparse[uint8]`` columns.
        Every other narwhals-supported backend (polars, pyarrow, ...) is encoded via
        ``narwhals.Series.to_dummies`` and returns **dense** integer columns,
        since those backends have no sparse-array equivalent.

        Parameters
        ----------
        X
            A dataframe where each column is a categorical feature.

        """
        X_nw = utils.dataframe.into_frame(X)
        native = (
            self._transform_many_pandas(typing.cast("pd.DataFrame", X))
            if X_nw.implementation.is_pandas()
            else self._transform_many_narwhals(X_nw)
        )
        return typing.cast("IntoDataFrameT", native)

    def _transform_many_pandas(self, X: pd.DataFrame) -> pd.DataFrame:
        pd = utils.pandas.import_pandas()
        oh = pd.get_dummies(X, columns=X.columns, sparse=True, dtype="uint8")

        # NOTE: assume if category mappings are explicitly provided,
        # no other category values are allowed for output. Aligns with `sklearn` behavior.
        if self.categories is not None:
            seen_in_the_past = {f"{col}_{val}" for col, vals in self.values.items() for val in vals}
            to_remove = set(oh.columns) - seen_in_the_past
            oh.drop(columns=list(to_remove), inplace=True)

        if not self.drop_zeros:
            seen_in_the_past = {f"{col}_{val}" for col, vals in self.values.items() for val in vals}
            to_add = seen_in_the_past - set(oh.columns)
            for col in to_add:
                oh[col] = pd.arrays.SparseArray([0] * len(oh), dtype="uint8")

        if self.drop_first:
            oh = oh.drop(columns=min(oh.columns))

        return oh

    def _transform_many_narwhals(self, X_nw: nw.DataFrame[IntoDataFrameT]) -> IntoDataFrameT:
        columns = X_nw.columns
        schema = X_nw.schema

        # `to_dummies` names columns `<feature>_<value>`, matching `transform_one`'s `f"{i}_{xi}"`
        # and the pandas `get_dummies` output. Null cells yield a `<feature>_null` column that
        # `pandas.get_dummies` omits, so they are dropped below for parity. `get_dummies` also omits
        # `NaN`; polars/pyarrow keep `NaN` distinct from null (it would otherwise leak a
        # `<feature>_NaN` dummy), so float `NaN` is folded into null first to match pandas exactly.
        dummies = nw.concat(
            [
                (X_nw[col].fill_nan(None) if schema[col].is_float() else X_nw[col]).to_dummies(
                    separator="_"
                )
                for col in columns
            ],
            how="horizontal",
        )
        null_cols = {f"{col}_null" for col in columns}
        present = {col for col in dummies.columns if col not in null_cols}
        seen_in_the_past = {f"{col}_{val}" for col, vals in self.values.items() for val in vals}

        # NOTE: assume if category mappings are explicitly provided, no other category values are
        # allowed for output. Aligns with `sklearn` behavior.
        keep = (present & seen_in_the_past) if self.categories is not None else set(present)
        if not self.drop_zeros:
            # Pad with the categories seen during past `learn_many` calls (zero columns).
            keep |= seen_in_the_past

        # Materialise the padded/null columns as all-zeros. They are added to `dummies` (which
        # still carries the row count) before any narrowing select, because selecting an empty set
        # of columns collapses the row count to zero on some backends (e.g. polars).
        to_add = keep - set(dummies.columns)
        zero_dtype = dummies.schema[dummies.columns[0]] if dummies.columns else nw.Int64
        oh = dummies.with_columns(
            [nw.lit(0, dtype=zero_dtype).alias(col) for col in sorted(to_add)]
        )

        final_columns = sorted(keep)
        if self.drop_first and final_columns:
            # Mirror the pandas path, which drops the lexicographically smallest column.
            final_columns = final_columns[1:]

        return oh.select(final_columns).to_native()
