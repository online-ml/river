from .. import base


class Discarder(base.Transformer):
    """Discards a given set of features.

    Parameters:
        black_list (str or list): Key(s) to delete.

    Example:

    ::

        >>> from creme import preprocessing

        >>> x = {'a': 42, 'b': 12}
        >>> preprocessing.Discarder(black_list=['a', 'zoidberg']).fit_one(x).transform_one(x)
        {'b': 12}

        >>> preprocessing.Discarder(black_list='b').fit_one(x).transform_one(x)
        {'a': 42}

    """

    def __init__(self, black_list=None):
        self.black_list = set(black_list if isinstance(black_list, list) else [black_list])

    def transform_one(self, x):
        return {i: x[i] for i in set(x.keys()) - self.black_list}
