from __future__ import annotations

import typing

from river import base

__all__ = ["FuncTransformer"]


class FuncTransformer(base.MiniBatchTransformer):
    """Wraps a function to make it usable in a pipeline.

    There is often a need to apply an arbitrary transformation to a set of features. For instance,
    this could involve parsing a date and then extracting the hour from said date. If you're
    processing a stream of data, then you can do this yourself by calling the necessary code at
    your leisure. On the other hand, if you want to do this as part of a pipeline, then you need to
    follow a simple convention.

    To use a function as part of a pipeline, take as input a `dict` of features and output a `dict`.
    Once you have initialized this class with your function, then you can use it like you would use
    any other (unsupervised) transformer.

    It is up to you if you want your function to be pure or not. By pure we refer to a function
    that doesn't modify its input. However, we recommend writing pure functions because this
    reduces the chances of inserting bugs into your pipeline.

    Parameters
    ----------
    func
        A function that takes as input a `dict` and outputs a `dict`.

    Examples
    --------

    >>> from pprint import pprint
    >>> import datetime as dt
    >>> from river import compose

    >>> x = {'date': '2019-02-14'}

    >>> def parse_date(x):
    ...     date = dt.datetime.strptime(x['date'], '%Y-%m-%d')
    ...     x['is_weekend'] = date.day in (5, 6)
    ...     x['hour'] = date.hour
    ...     return x

    >>> t = compose.FuncTransformer(parse_date)
    >>> pprint(t.transform_one(x))
    {'date': '2019-02-14', 'hour': 0, 'is_weekend': False}

    The above example is not pure because it modifies the input. The following example is pure
    and produces the same output:

    >>> def parse_date(x):
    ...     date = dt.datetime.strptime(x['date'], '%Y-%m-%d')
    ...     return {'is_weekend': date.day in (5, 6), 'hour': date.hour}

    >>> t = compose.FuncTransformer(parse_date)
    >>> pprint(t.transform_one(x))
    {'hour': 0, 'is_weekend': False}

    The previous example doesn't include the `date` feature because it returns a new `dict`.
    However, a common usecase is to add a feature to an existing set of features. You can do
    this in a pure way by unpacking the input `dict` into the output `dict`:

    >>> def parse_date(x):
    ...     date = dt.datetime.strptime(x['date'], '%Y-%m-%d')
    ...     return {'is_weekend': date.day in (5, 6), 'hour': date.hour, **x}

    >>> t = compose.FuncTransformer(parse_date)
    >>> pprint(t.transform_one(x))
    {'date': '2019-02-14', 'hour': 0, 'is_weekend': False}

    You can add `FuncTransformer` to a pipeline just like you would with any other transformer.

    >>> from river import naive_bayes

    >>> pipeline = compose.FuncTransformer(parse_date) | naive_bayes.MultinomialNB()
    >>> pipeline
    Pipeline (
      FuncTransformer (
        func="parse_date"
      ),
      MultinomialNB (
        alpha=1.
      )
    )

    If you provide a function without wrapping it, then the pipeline will do it for you:

    >>> pipeline = parse_date | naive_bayes.MultinomialNB()

    """

    def __init__(self, func: typing.Callable[[dict], dict]):
        self.func = func

    def transform_one(self, x):
        return self.func(x)

    def transform_many(self, X):
        return self.func(X)

    def __str__(self):
        return self.func.__name__
