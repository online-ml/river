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
            >>> compose.Whitelister('a').transform_one(x)
            {'a': 42}

    """

    def __init__(self, *whitelist):
        self.whitelist = whitelist

    def transform_one(self, x):
        return {i: x[i] for i in self.whitelist}

    def __str__(self):
        return str(self.whitelist)
