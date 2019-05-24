from .. import base


__all__ = ['FuncTransformer']


class FuncTransformer(base.Transformer):
    """Transforms a set of features following a given callable.

    The provided function has to take as input a ``dict`` of features and produce a new ``dict`` of
    transformed features.

    Parameters:
        func (callable): a function transforming a dict of features into a new dict of features.

    Example:

        ::

            >>> import datetime as dt
            >>> from creme import compose

            >>> x = {'date': '2019-02-14', 'x': 42}

            >>> def parse_date(x):
            ...     x['date'] = dt.datetime.strptime(x['date'], '%Y-%m-%d')
            ...     x['is_weekend'] = x['date'].day in (5, 6)
            ...     return x

            >>> transformer = compose.FuncTransformer(parse_date)
            >>> transformer.transform_one(x)
            {'date': datetime.datetime(2019, 2, 14, 0, 0), 'x': 42, 'is_weekend': False}

    """

    def __init__(self, func):
        self.func = func

    def transform_one(self, x):
        return self.func(x)

    def __str__(self):
        return self.func.__name__
