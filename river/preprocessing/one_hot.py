import collections

import pandas as pd
import numpy as np

from river import base

__all__ = ["OneHotEncoder"]


class OneHotEncoder(base.Transformer):
    """One-hot encoding.

    This transformer will encode every feature it is provided with.
    If a list or set is provided, this transformer will encode every entry in the list/set.
    You can apply it to a subset of features by composing it
     with `compose.Select` or `compose.SelectType`.

    Parameters
    ----------
    sparse
        Whether or not 0s should be made explicit or not.

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

    >>> import river.preprocessing

    >>> oh = river.preprocessing.OneHotEncoder(sparse=True)
    >>> for x in X:
    ...     oh = oh.learn_one(x)
    ...     pprint(oh.transform_one(x))
    {'c1_u': 1, 'c2_d': 1}
    {'c1_a': 1, 'c2_x': 1}
    {'c1_i': 1, 'c2_h': 1}
    {'c1_h': 1, 'c2_e': 1}

    The `sparse` parameter can be set to `False` in order to include the values that are not
    present in the output.

    >>> oh = river.preprocessing.OneHotEncoder(sparse=False)
    >>> for x in X[:2]:
    ...     oh = oh.learn_one(x)
    ...     pprint(oh.transform_one(x))
    {'c1_u': 1, 'c2_d': 1}
    {'c1_a': 1, 'c1_u': 0, 'c2_d': 0, 'c2_x': 1}

    A subset of the features can be one-hot encoded by using an instance of `compose.Select`.

    >>> from river import compose

    >>> pp = compose.Select('c1') | river.preprocessing.OneHotEncoder()

    >>> for x in X:
    ...     pp = pp.learn_one(x)
    ...     pprint(pp.transform_one(x))
    {'c1_u': 1}
    {'c1_a': 1, 'c1_u': 0}
    {'c1_a': 0, 'c1_i': 1, 'c1_u': 0}
    {'c1_a': 0, 'c1_h': 1, 'c1_i': 0, 'c1_u': 0}

    You can preserve the `c2` feature by using a union:

    >>> pp = compose.Select('c1') | river.preprocessing.OneHotEncoder()
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

    >>> oh = river.preprocessing.OneHotEncoder(sparse=True)
    >>> for x in X:
    ...     oh = oh.learn_one(x)
    ...     pprint(oh.transform_one(x))
    {'c1_a': 1, 'c1_u': 1, 'c2_d': 1}
    {'c1_a': 1, 'c1_b': 1, 'c2_x': 1}
    {'c1_i': 1, 'c2_h': 1, 'c2_z': 1}
    {'c1_b': 1, 'c1_h': 1, 'c2_e': 1}

    Mini-batching

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

    >>> import river.preprocessing

    >>> oh = river.preprocessing.OneHotEncoder(sparse=True)
    >>> oh = oh.learn_many(pd.DataFrame(X))

    >>> oh.transform_many(pd.DataFrame(X))
        c1_h  c1_u  c1_i  c1_a  c2_x  c2_h  c2_e  c2_d
    0     0     1     0     0     0     0     0     1
    1     0     0     0     1     1     0     0     0
    2     0     0     1     0     0     1     0     0
    3     1     0     0     0     0     0     1     0

    Keep in mind that ability for sparse transformations is limited in mini-batch case,
    which might affect speed/memory footprint of your training loop.

    Here's a non-sparse example:
    >>> oh = river.preprocessing.OneHotEncoder(sparse=False)
    >>> X_init = [{"c1": "Oranges", "c2": "Apples"}]
    >>> oh = oh.learn_many(pd.DataFrame(X_init))
    >>> oh = oh.learn_many(pd.DataFrame(X))

    >>> oh.transform_many(X=pd.DataFrame(X))
        c1_h  c1_i  c1_Oranges  c1_u  c1_a  c2_h  c2_x  c2_e  c2_d  c2_Apples
    0     0     0           0     1     0     0     0     0     1          0
    1     0     0           0     0     1     0     1     0     0          0
    2     0     1           0     0     0     1     0     0     0          0
    3     1     0           0     0     0     0     0     1     0          0

    """

    def __init__(self, sparse=False):
        self.sparse = sparse
        self.values = collections.defaultdict(set)

    def learn_one(self, x):
        for i, xi in x.items():
            if isinstance(xi, (list, set)):
                for xj in xi:
                    self.values[i].add(xj)
            else:
                self.values[i].add(xi)

        return self

    def transform_one(self, x, y=None):
        oh = {}

        # Add 0s
        if not self.sparse:
            oh = {f"{i}_{v}": 0 for i, values in self.values.items() for v in values}

        # Add 1s
        for i, xi in x.items():
            if isinstance(xi, (list, set)):
                for xj in xi:
                    oh[f"{i}_{xj}"] = 1
            else:
                oh[f"{i}_{xi}"] = 1

        return oh

    # Mini-batch methods

    @staticmethod
    def _encode_1d(data, prefix, categories=None, sparse=False):
        # REFLECTION: river often descends onto numpy level for computations
        # and just carries column names around, maybe try the same on next iteration
        # INFO: inspired by:
        # https://github.com/pandas-dev/pandas/blob/66e3805b8cabe977f40c05259cc3fcf7ead5687d/pandas/core/reshape/reshape.py#L936

        if categories is not None:
            if type(categories) is not set:
                categories = set(categories)

            if sparse:
                categories = categories & set(data)

            cat = pd.Categorical(data, categories=categories, ordered=False)
        else:
            cat = pd.Categorical(data, ordered=False)
        categories = cat.categories
        # cat.add_categories INFO: look into this for learninig?

        codes = cat.codes

        number_of_cols = len(categories)
        dummy_mat = np.eye(number_of_cols, dtype=int).take(codes, axis=1).T
        # reset NaN GH4446[pandas]
        dummy_mat[codes == -1] = 0

        return pd.DataFrame(
            dummy_mat, index=data.index, columns=[f"{prefix}_{v}" for v in categories]
        )

    def learn_many(self, X: pd.DataFrame):

        for col in X.columns:
            self.values[col].update(set(X.loc[:, col]))

        return self

    def transform_many(self, X: pd.DataFrame):

        Xt = list()

        for col, values in self.values.items():
            xt = self._encode_1d(
                data=X.loc[:, col], prefix=col, categories=values, sparse=self.sparse
            )
            Xt.append(xt)

        # INFO: otherwise throws error if nothing to concatenate
        # (when inferring from blank state stansformer during learning, which is done inside `Pipeline`)
        if len(Xt) == 0:
            return pd.DataFrame(index=X.index, copy=False)
        else:
            return pd.concat(Xt, axis=1, copy=False)
