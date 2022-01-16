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

    # @staticmethod
    def _encode_1d(self, data, prefix, categories=None, sparse=False):
        # REFLECTION: river often descends onto numpy level for computations
        # and just carries column names around, maybe try the same on next iteration
        # INFO: inspired by:
        # https://github.com/pandas-dev/pandas/blob/66e3805b8cabe977f40c05259cc3fcf7ead5687d/pandas/core/reshape/reshape.py#L936

        # cat = pd.Categorical(data.iloc[:, 0], ordered=False)
        # cat = pd.Categorical(data.iloc[:, 0], categories=["a", "h"], ordered=False)
        # cat = pd.Categorical(data.iloc[:, :], categories=["a", "h"], ordered=False) # INFO: won't work
        # cat = pd.Categorical(data, categories=["a", "h"], ordered=False)

        if categories is not None:
            # INFO: give up guarantees of any ordering - to align behaviour with sparse implementation
            # REFLECTION: guarantees of ordering are necessary for proper gradient updates it seems,
            # coz gradients don't map to column names arbitrarily reordered
            # UPD: above is wrong, order guarantees are not necessary, see GH818
            if type(categories) is not set:
                categories = set(categories)

            if sparse:
                categories = categories & set(data)

            cat = pd.Categorical(data, categories=categories, ordered=False)
        else:
            cat = pd.Categorical(data, ordered=False)
        categories = cat.categories
        # cat.add_categories INFO: look into this for learninig?
        print(categories)
        codes = cat.codes
        print(codes)

        # if sparse:
        # categories = categories[codes != -1]
        # print(categories)
        # print(categories[codes != -1])
        # categories = categories[:2]
        # categories = categories[: np.max(codes) + 1]
        # categories = categories[categories.isin(np.unique(codes))]
        # categories = categories[np.unique(codes)]

        # print(categories)
        # codes = codes[codes != -1]
        # print(codes)

        number_of_cols = len(categories)
        dummy_mat = np.eye(number_of_cols, dtype=int).take(codes, axis=1).T
        # reset NaN GH4446[pandas]
        dummy_mat[codes == -1] = 0

        # return pd.DataFrame(dummy_mat, index=data.index, columns=categories)
        return pd.DataFrame(
            dummy_mat, index=data.index, columns=[f"{prefix}_{v}" for v in categories]
        )
        # REFLECTION: wanted to reuse this func for learn_many
        # but the new modality with the code below is only really necessary for learn_transform scenario
        # if not output_categories:
        #     return pd.DataFrame(dummy_mat, index=data.index, columns=[f"{prefix}_{v}" for v in categories])
        # else:
        #     return pd.DataFrame(dummy_mat, index=data.index, columns=[f"{prefix}_{v}" for v in categories]), categories

    def learn_many(self, X: pd.DataFrame):

        # print("OneHotEncoderExtended - learn_many!")
        # TODO: add testing
        # pass
        # Xt = list()
        # for xi, _ in stream.iter_pandas(X, None):
        #     # print(xi)
        #     # xi_t = self.learn_one(xi)
        #     self = self.learn_one(xi)
        #     # Xt.append(xi_t)
        # # return Xt

        for col in X.columns:
            # print(col)
            self.values[col].update(set(X.loc[:, col]))
            # print(self.values[col])

        return self

    def transform_many(self, X: pd.DataFrame):
        # print('JELLO')
        # FIXME: for some reason calls to transform_many instead of learn_many is made when calling learn_many on a Pipeline object
        # print("OneHotEncoderExtended - transform_many!")
        # index = X.index
        # pass
        # if not self.sparse:
        # Xt = list()
        # for xi, _ in stream.iter_pandas(X, None):
        #     # print(xi)
        #     xi_t = self.transform_one(xi)
        #     Xt.append(xi_t)
        # # return Xt
        # return pd.DataFrame(Xt, copy=False)
        Xt = list()
        # print(self.values)
        for col, values in self.values.items():
            # FIXME: check, this code path is probably never reached
            # print(f"constructing transformation for column {col}")
            # print(col)
            # print(X.loc[:, col])
            # print(X.shape)
            # print(X.loc[:, col].shape)
            # # display(X.loc[:, col])
            # print(type(X.loc[:, col]))
            # print(type(values))
            # values.update(["ff"]) # INFO: for testing sparsity
            # xt = self._encode_1d(data=X.loc[:, col], prefix=col, categories=values)
            xt = self._encode_1d(
                data=X.loc[:, col], prefix=col, categories=values, sparse=self.sparse
            )
            # xt = self._encode_1d(X.loc[:, col], col, values, self.sparse)
            # break
            Xt.append(xt)
        # return xt
        # FIXME: throws error if nothing to concatenate (when inferring from blank state stansformer)
        # print(Xt)
        # if Xt.isempty():
        if len(Xt) == 0:
            return pd.DataFrame(index=X.index, copy=False)
        else:
            return pd.concat(Xt, axis=1, copy=False)
        # else:
        #     # TODO: add testing
        #     # raise NotImplementedError("Only sparse=False is supported for now")
        #     # Xt = list()
        #     # for xi, _ in stream.iter_pandas(X, None):
        #     #     # print(xi)
        #     #     xi_t = self.transform_one(xi)
        #     #     Xt.append(xi_t)
        #     # # return Xt
        #     # # return pd.DataFrame(Xt, copy=False)

        #     # # INFO: otherwise it'll be floats + NaNs
        #     # return pd.DataFrame(Xt, copy=False, index=index).fillna(0).astype(np.int0)
        #     Xt = list()
        #     for col, values in self.values.items():
        #         print(col)
        #         print(X.loc[:, col])
        #         # xt = self._encode_1d(data=X.loc[:, col], prefix=col, categories=values)
        #         xt = _encode_1d(data=X.loc[:, col], prefix=col, categories=values, sparse)
        #         # break
        #         Xt.append(xt)
        #     # return xt
        #     return pd.concat(Xt, axis=1)
        # # return X.copy()
