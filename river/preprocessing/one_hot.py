from __future__ import annotations

import collections

import pandas as pd

from river import base

__all__ = ["OneHotEncoder"]


class OneHotEncoder(base.MiniBatchTransformer):
    """One-hot encoding.

    This transformer will encode every feature it is provided with.
    If a list or set is provided, this transformer will encode every entry in the list/set.
    You can apply it to a subset of features by composing it
     with `compose.Select` or `compose.SelectType`.

    Parameters
    ----------
    drop_zeros
        Whether or not 0s should be made explicit or not.
    drop_first
        Whether to get `k - 1` dummies out of `k` categorical levels by removing the first key.
        This is useful in some statistical models where perfectly collinear features cause
        problems.

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

    e can now apply one-hot encoding. All the provided are one-hot encoded, there is therefore
    no need to specify which features to encode.

    >>> from river import preprocessing

    >>> oh = preprocessing.OneHotEncoder()
    >>> for x in X[:2]:
    ...     oh = oh.learn_one(x)
    ...     pprint(oh.transform_one(x))
    {'c1_u': 1, 'c2_d': 1}
    {'c1_a': 1, 'c1_u': 0, 'c2_d': 0, 'c2_x': 1}

    The `drop_zeros` parameter can be set to `True` if you don't want the past features to be included
    in the output. Otherwise, all the past features will be included in the output.

    >>> oh = preprocessing.OneHotEncoder(drop_zeros=True)
    >>> for x in X:
    ...     oh = oh.learn_one(x)
    ...     pprint(oh.transform_one(x))
    {'c1_u': 1, 'c2_d': 1}
    {'c1_a': 1, 'c2_x': 1}
    {'c1_i': 1, 'c2_h': 1}
    {'c1_h': 1, 'c2_e': 1}

    You can encode only `k - 1` features out of `k` by setting `drop_first` to `True`.

    >>> oh = preprocessing.OneHotEncoder(drop_first=True, drop_zeros=True)
    >>> for x in X:
    ...     oh = oh.learn_one(x)
    ...     pprint(oh.transform_one(x))
    {'c2_d': 1}
    {'c2_x': 1}
    {'c2_h': 1}
    {'c2_e': 1}

    A subset of the features can be one-hot encoded by piping a `compose.Select` into the
    `OneHotEncoder`.

    >>> from river import compose

    >>> pp = compose.Select('c1') | preprocessing.OneHotEncoder()

    >>> for x in X:
    ...     pp = pp.learn_one(x)
    ...     pprint(pp.transform_one(x))
    {'c1_u': 1}
    {'c1_a': 1, 'c1_u': 0}
    {'c1_a': 0, 'c1_i': 1, 'c1_u': 0}
    {'c1_a': 0, 'c1_h': 1, 'c1_i': 0, 'c1_u': 0}

    You can preserve the `c2` feature by using a union:

    >>> pp = compose.Select('c1') | preprocessing.OneHotEncoder()
    >>> pp += compose.Select('c2')

    >>> for x in X:
    ...     pp = pp.learn_one(x)
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
    ...     oh = oh.learn_one(x)
    ...     pprint(oh.transform_one(x))
    {'c1_a': 1, 'c1_u': 1, 'c2_d': 1}
    {'c1_a': 1, 'c1_b': 1, 'c2_x': 1}
    {'c1_i': 1, 'c2_h': 1, 'c2_z': 1}
    {'c1_b': 1, 'c1_h': 1, 'c2_e': 1}

    Processing mini-batches is also possible.

    >>> from pprint import pprint
    >>> import random
    >>> import string

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
    >>> oh = oh.learn_many(X_init)
    >>> oh = oh.learn_many(X)

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

    """

    def __init__(self, drop_zeros=False, drop_first=False):
        self.drop_zeros = drop_zeros
        self.drop_first = drop_first
        self.values = collections.defaultdict(set)

    def learn_one(self, x):
        if self.drop_zeros:
            return self

        for i, xi in x.items():
            if isinstance(xi, list) or isinstance(xi, set):
                for xj in xi:
                    self.values[i].add(xj)
            else:
                self.values[i].add(xi)

        return self

    def transform_one(self, x, y=None):
        oh = {}

        # Add 0s
        if not self.drop_zeros:
            oh = {f"{i}_{v}": 0 for i, values in self.values.items() for v in values}

        # Add 1s
        for i, xi in x.items():
            if isinstance(xi, list) or isinstance(xi, set):
                for xj in xi:
                    oh[f"{i}_{xj}"] = 1
            else:
                oh[f"{i}_{xi}"] = 1

        if self.drop_first:
            oh.pop(min(oh.keys()))

        return oh

    def learn_many(self, X):
        if self.drop_zeros:
            return self

        for col in X.columns:
            self.values[col].update(X[col].unique())

        return self

    def transform_many(self, X):
        oh = pd.get_dummies(X, columns=X.columns, sparse=True, dtype="uint8")

        if not self.drop_zeros:
            seen_in_the_past = {f"{col}_{val}" for col, vals in self.values.items() for val in vals}
            to_add = seen_in_the_past - set(oh.columns)
            for col in to_add:
                oh[col] = pd.arrays.SparseArray([0] * len(oh), dtype="uint8")

        if self.drop_first:
            oh = oh.drop(columns=min(oh.columns))

        return oh
