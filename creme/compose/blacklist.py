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
            >>> compose.Blacklister('a', 'zoidberg').transform_one(x)
            {'b': 12}

            >>> compose.Blacklister('b').transform_one(x)
            {'a': 42}

    """

    def __init__(self, *blacklist):
        self.blacklist = set(blacklist)

    def transform_one(self, x):
        return {i: x[i] for i in set(x.keys()) - self.blacklist}

    def __str__(self):
        return '~' + str(sorted(self.blacklist))
