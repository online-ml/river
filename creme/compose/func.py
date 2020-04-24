import typing

from creme import base
from creme import stats


__all__ = ['FuncTransformer', 'FuncStat']


class FuncTransformer(base.Transformer):
    """Transforms a set of features following a given callable.

    The provided function has to take as input a `dict` of features and produce a new `dict` of
    transformed features.

    Parameters:
        func: A function transforming a dict of features into a new dict of features.

    Example:

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

    def __init__(self, func: typing.Callable[[dict], dict]):
        self.func = func

    def transform_one(self, x):
        return self.func(x)

    def __str__(self):
        return self.func.__name__


class FuncStat(stats.base.Statistic):
    """Transforms a set of features following a given callable.

    The provided function has to take as input a `dict` of features and produce a new `dict` of
    transformed features.

    Parameters:
        func: A function transforming a dict of features into a new dict of features.

    Example:

        >>> import datetime as dt
        >>> from creme import compose

        >>> x = '2019-02-14'

        >>> def parse_date(x):
        ...     x = dt.datetime.strptime(x, '%Y-%m-%d')
        ...     return x.day

        >>> stat = compose.FuncStat(parse_date)
        >>> stat.update(x)
        14

    """

    def __init__(self, func: typing.Callable[[dict], dict]):
        self.func = func

    def update(self, x):
        return self.func(x)

    def __str__(self):
        return self.func.__name__