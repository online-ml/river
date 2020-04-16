from .. import base


__all__ = ['Discard', 'Select']


class Discard(base.Transformer):
    """Subsets a set of features by applying a blacklist.

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
    """Subsets a set of features by applying a whitelist.

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
