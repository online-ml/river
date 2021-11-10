import typing

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

    def __init__(self, *keys: typing.Tuple[base.typing.FeatureName]):
        self.keys = set(keys)

    def transform_one(self, x):
        return {i: xi for i, xi in x.items() if i not in self.keys}

    def __str__(self):
        return "~" + str(sorted(self.keys))

    def __repr__(self):
        if self.keys:
            return "Discard (\n  " + "\n  ".join(map(str, sorted(self.keys))) + "\n)"
        return "Discard ()"

    def _set_params(self, keys=None):
        if not keys:
            keys = self.keys
        return Discard(*keys)


class Select(base.Transformer):
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

    """

    def __init__(self, *keys: typing.Tuple[base.typing.FeatureName]):
        self.keys = set(keys)

    def transform_one(self, x):
        return {i: x[i] for i in self.keys}

    def __str__(self):
        return str(sorted(self.keys))

    def __repr__(self):
        if self.keys:
            return "Select (\n  " + "\n  ".join(map(str, sorted(self.keys))) + "\n)"
        return "Select ()"

    def _get_params(self):
        return self.keys

    def _set_params(self, keys=None):
        if not keys:
            keys = self.keys
        return Select(*keys)


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

    def __init__(self, *types: typing.Tuple[type]):
        self.types = types

    def transform_one(self, x):
        return {i: xi for i, xi in x.items() if isinstance(xi, self.types)}

    def __str__(self):
        return f'Select({", ".join(t.__name__ for t in self.types)})'

    def __repr__(self):
        if self.types:
            return "Select (\n  " + "\n  ".join(map(str, sorted(self.types))) + "\n)"
        return "Select ()"

    def _get_params(self):
        return self.types

    def _set_params(self, types=None):
        if not types:
            types = self.types
        return SelectType(*types)
