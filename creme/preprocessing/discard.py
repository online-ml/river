from .. import base


class Discarder(base.Transformer):
    """Discards a given set of features.

    Parameters:
        black_list (str or list): Key(s) to delete.

    Example:

    ::

        >>> from creme import preprocessing

        >>> x = {'a': 42, 'b': 12, 'c': 2}
        >>> preprocessing.Discarder(black_list=['a', 'b', 'zoidberg']).fit_one(x)
        {'c': 2}

        >>> preprocessing.Discarder(black_list='c').fit_one(x)
        {'a': 42, 'b': 12}

    """

    def __init__(self, black_list=None):
        self.black_list = set(black_list if isinstance(black_list, list) else [black_list])

    def fit_one(self, x, y=None):
        return self.transform_one(x)

    def transform_one(self, x):
        return {i: x[i] for i in set(x.keys()) - self.black_list}
