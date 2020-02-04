from .. import base


__all__ = ['Blacklister']


class Blacklister(base.Transformer):
    """Subsets a set of features by applying a blacklist.

    Parameters:
        blacklist (strs): Key(s) to discard.

    Example:

        ::

            >>> from creme import compose

            >>> x = {'a': 42, 'b': 12}
            >>> compose.Blacklister('a').transform_one(x)
            {'b': 12}

    """

    def __init__(self, *blacklist):
        self.blacklist = blacklist

    def transform_one(self, x):
        return {i: xi for i, xi in x.items() if i not in self.blacklist}

    def __str__(self):
        return '~' + str(self.blacklist)
