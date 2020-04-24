import typing

from creme import base


__all__ = ['Discard', 'Select', 'SelectType']


class Discard(base.Transformer):
    """Removes features according to a blacklist.

    Parameters:
        blacklist: Key(s) to discard.

    Example:

        >>> from creme import compose

        >>> x = {'a': 42, 'b': 12, 'c': 13}
        >>> compose.Discard('a', 'b').transform_one(x)
        {'c': 13}

    """

    def __init__(self, *blacklist):
        self.blacklist = set(blacklist)

    def transform_one(self, x):
        return {i: xi for i, xi in x.items() if i not in self.blacklist}

    def __str__(self):
        return '~' + str(sorted(self.blacklist))

    def __repr__(self):
        if self.blacklist:
            return 'Discard (\n  ' + '\n  '.join(map(str, sorted(self.blacklist))) + '\n)'
        return 'Discard ()'

    def _set_params(self, blacklist=None):
        if not blacklist:
            blacklist = self.blacklist
        return Select(*blacklist)


class Select(base.Transformer):
    """Selects features according to a whitelist.

    Parameters:
        whitelist: Key(s) to keep.

    Example:

        >>> from creme import compose

        >>> x = {'a': 42, 'b': 12, 'c': 13}
        >>> compose.Select('c').transform_one(x)
        {'c': 13}

    """

    def __init__(self, *whitelist):
        self.whitelist = set(whitelist)

    def transform_one(self, x):
        return {i: x[i] for i in self.whitelist}

    def __str__(self):
        return str(sorted(self.whitelist))

    def __repr__(self):
        if self.whitelist:
            return 'Select (\n  ' + '\n  '.join(map(str, sorted(self.whitelist))) + '\n)'
        return 'Select ()'

    def _get_params(self):
        return self.whitelist

    def _set_params(self, whitelist=None):
        if not whitelist:
            whitelist = self.whitelist
        return Select(*whitelist)


class SelectType(base.Transformer):
    """Selects features based on their type.

    This is practical when you want to apply different preprocessing steps to different kinds of
    features. For instance, a common usecase is to apply a `preprocessing.StandardScaler` to
    numeric features and a `preprocessing.OneHotEncoder` to categorical features.

    Example:

        >>> import numbers
        >>> from creme import compose
        >>> from creme import linear_model
        >>> from creme import preprocessing

        >>> num = compose.SelectType(numbers.Number) | preprocessing.StandardScaler()
        >>> cat = compose.SelectType(str) | preprocessing.OneHotEncoder()
        >>> model = (num + cat) | linear_model.LogisticRegression()

    """

    def __init__(self, *types: typing.Tuple[type]):
        self.types = types

    def transform_one(self, x):
        return {i: xi for i, xi in x.items() if isinstance(xi, self.types)}

    def _get_params(self):
        return self.types

    def _set_params(self, types=None):
        if not types:
            types = self.types
        return SelectType(*types)
