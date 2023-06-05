from __future__ import annotations

from river import base

__all__ = ["Discard", "Select", "SelectType"]


class Discard(base.Transformer):
    """Removes features.

    This can be used in a pipeline when you want to remove certain features. The `transform_one`
    method is pure, and therefore returns a fresh new dictionary instead of removing the specified
    keys from the input.

    Parameters
    ----------
    keys
        Key(s) to discard.

    Examples
    --------

    >>> from river import compose

    >>> x = {'a': 42, 'b': 12, 'c': 13}
    >>> compose.Discard('a', 'b').transform_one(x)
    {'c': 13}

    You can chain a discarder with any estimator in order to apply said estimator to the
    desired features.

    >>> from river import feature_extraction as fx

    >>> x = {'sales': 10, 'shop': 'Ikea', 'country': 'Sweden'}

    >>> pipeline = (
    ...     compose.Discard('shop', 'country') |
    ...     fx.PolynomialExtender()
    ... )
    >>> pipeline.transform_one(x)
    {'sales': 10, 'sales*sales': 100}

    """

    def __init__(self, *keys: tuple[base.typing.FeatureName]):
        self.keys = set(keys)

    def transform_one(self, x):
        return {i: xi for i, xi in x.items() if i not in self.keys}

    def __str__(self):
        return "~" + str(sorted(self.keys))

    def __repr__(self):
        if self.keys:
            return "Discard (\n  " + "\n  ".join(map(str, sorted(self.keys))) + "\n)"
        return "Discard ()"


class Select(base.MiniBatchTransformer):
    """Selects features.

    This can be used in a pipeline when you want to select certain features. The `transform_one`
    method is pure, and therefore returns a fresh new dictionary instead of filtering the specified
    keys from the input.

    Parameters
    ----------
    keys
        Key(s) to keep.

    Examples
    --------

    >>> from river import compose

    >>> x = {'a': 42, 'b': 12, 'c': 13}
    >>> compose.Select('c').transform_one(x)
    {'c': 13}

    You can chain a selector with any estimator in order to apply said estimator to the
    desired features.

    >>> from river import feature_extraction as fx

    >>> x = {'sales': 10, 'shop': 'Ikea', 'country': 'Sweden'}

    >>> pipeline = (
    ...     compose.Select('sales') |
    ...     fx.PolynomialExtender()
    ... )
    >>> pipeline.transform_one(x)
    {'sales': 10, 'sales*sales': 100}

    This transformer also supports mini-batch processing:

    >>> import random
    >>> from river import compose

    >>> random.seed(42)
    >>> X = [{"x_1": random.uniform(8, 12), "x_2": random.uniform(8, 12)} for _ in range(6)]
    >>> for x in X:
    ...     print(x)
    {'x_1': 10.557707193831535, 'x_2': 8.100043020890668}
    {'x_1': 9.100117273476478, 'x_2': 8.892842952595291}
    {'x_1': 10.94588485665605, 'x_2': 10.706797949691644}
    {'x_1': 11.568718270819382, 'x_2': 8.347755330517664}
    {'x_1': 9.687687278741082, 'x_2': 8.119188877752281}
    {'x_1': 8.874551899214413, 'x_2': 10.021421152413449}

    >>> import pandas as pd
    >>> X = pd.DataFrame.from_dict(X)

    You can then call `transform_many` to transform a mini-batch of features:

    >>> compose.Select('x_2').transform_many(X)
        x_2
    0   8.100043
    1   8.892843
    2  10.706798
    3   8.347755
    4   8.119189
    5  10.021421

    """

    def __init__(self, *keys: tuple[base.typing.FeatureName]):
        self.keys = set(keys)

    def transform_one(self, x):
        return {i: x[i] for i in self.keys}

    def transform_many(self, X):
        # INFO: has either side-effects or doesn't have copy - choose your poison
        # REFLECTION: worth adding `copy=True` parameter to the object constructor to allow both?
        # << convention is to have pure methods/functions
        return X.loc[:, list(self.keys)].copy()
        # return X.loc[:, self.keys]

    def __str__(self):
        return str(sorted(self.keys))

    def __repr__(self):
        if self.keys:
            return "Select (\n  " + "\n  ".join(map(str, sorted(self.keys))) + "\n)"
        return "Select ()"


class SelectType(base.Transformer):
    """Selects features based on their type.

    This is practical when you want to apply different preprocessing steps to different kinds of
    features. For instance, a common usecase is to apply a `preprocessing.StandardScaler` to
    numeric features and a `preprocessing.OneHotEncoder` to categorical features.

    Parameters
    ----------
    types
        Python types which you want to select. Under the hood, the `isinstance` method will be used
        to check if a value is of a given type.

    Examples
    --------

    >>> import numbers
    >>> from river import compose
    >>> from river import linear_model
    >>> from river import preprocessing

    >>> num = compose.SelectType(numbers.Number) | preprocessing.StandardScaler()
    >>> cat = compose.SelectType(str) | preprocessing.OneHotEncoder()
    >>> model = (num + cat) | linear_model.LogisticRegression()

    """

    def __init__(self, *types: tuple[type]):
        self.types = types

    def transform_one(self, x):
        return {i: xi for i, xi in x.items() if isinstance(xi, self.types)}

    def __str__(self):
        return f'Select({", ".join(t.__name__ for t in self.types)})'

    def __repr__(self):
        if self.types:
            return "Select (\n  " + "\n  ".join(map(str, sorted(self.types))) + "\n)"
        return "Select ()"
