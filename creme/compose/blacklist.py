from .. import base


__all__ = ['Blacklister']


class Blacklister(base.Transformer):
    """Subsets a set of features by applying a blacklist.

    Parameters:
        blacklist (str or list): Key(s) to discard.

    Example:

        ::

            >>> from creme import compose

            >>> x = {'a': 42, 'b': 12}
            >>> compose.Blacklister(['a', 'zoidberg']).transform_one(x)
            {'b': 12}

            >>> compose.Blacklister('b').transform_one(x)
            {'a': 42}

    """

    def __init__(self, blacklist=None):
        if blacklist is None:
            blacklist = []
        self.blacklist = set(blacklist if isinstance(blacklist, (list, tuple)) else [blacklist])

    def transform_one(self, x):
        return {i: x[i] for i in set(x.keys()) - self.blacklist}

    def __str__(self):
        return self.__class__.__name__ + f'({list(self.blacklist)})'
