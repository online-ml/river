from .. import base


__all__ = ['Whitelister']


class Whitelister(base.Transformer):
    """Subsets a set of features by applying a whitelist.

    Parameters:
        whitelist (strs): Key(s) to keep.

    Example:

        ::

            >>> from creme import compose

            >>> x = {'a': 42, 'b': 12}
            >>> compose.Whitelister('a', 'zoidberg').transform_one(x)
            {'a': 42}

            >>> compose.Whitelister('b').transform_one(x)
            {'b': 12}

    """

    def __init__(self, *whitelist):
        self.whitelist = set(whitelist)

    def transform_one(self, x):
        return {i: x[i] for i in set(x.keys()) & self.whitelist}

    def __str__(self):
        return str(sorted(self.whitelist))
