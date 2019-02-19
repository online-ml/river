from .. import base


class FuncExtractor(base.Transformer):
    """Extracts one or more feature(s) following a given callable.

    The provided function has to take as input a ``dict`` of features and produce a new ``dict`` of
    computed features.

    Example:

    ::

        >>> import datetime as dt
        >>> from creme import feature_extraction

        >>> x = {'date': '2019-02-14', 'x': 42}

        >>> def is_weekend(x):
        ...     date = dt.datetime.strptime(x['date'], '%Y-%m-%d')
        ...     return {'is_weekend': date.day in (5, 6)}

        >>> extractor = feature_extraction.FuncExtractor(is_weekend)
        >>> extractor.fit_one(x)
        {'is_weekend': False}

    """

    def __init__(self, func):
        self.func = func

    def fit_one(self, x, y=None):
        return self.transform_one(x)

    def transform_one(self, x):
        return self.func(x)
