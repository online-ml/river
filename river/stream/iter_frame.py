from __future__ import annotations

import typing

import narwhals.stable.v2 as nw

from river import base, stream

if typing.TYPE_CHECKING:
    from narwhals.stable.v2.typing import IntoDataFrame, IntoSeries


def iter_frame(
    X: IntoDataFrame, y: IntoSeries | IntoDataFrame | None = None, **kwargs
) -> base.typing.Stream:
    """Iterates over the rows of a dataframe.

    This is a dataframe-agnostic iterator: it works with any eager dataframe supported by
    [Narwhals](https://narwhals-dev.github.io/narwhals/) (pandas, polars, PyArrow, Modin, cuDF,
    ...). It supersedes `stream.iter_pandas` and `stream.iter_polars`.

    Parameters
    ----------
    X
        A dataframe of features. Any eager dataframe supported by Narwhals will work.
    y
        A series or a dataframe with one column per target.
    kwargs
        Extra keyword arguments are passed to the underlying call to `stream.iter_array`.

    Examples
    --------

    >>> from river import stream

    The same code works regardless of the dataframe library. With pandas:

    >>> import pandas as pd
    >>> X = pd.DataFrame({
    ...     'x1': [1, 2, 3, 4],
    ...     'x2': ['blue', 'yellow', 'yellow', 'blue'],
    ...     'y': [True, False, False, True]
    ... })
    >>> y = X.pop('y')

    >>> for xi, yi in stream.iter_frame(X, y):
    ...     print(xi, yi)
    {'x1': 1, 'x2': 'blue'} True
    {'x1': 2, 'x2': 'yellow'} False
    {'x1': 3, 'x2': 'yellow'} False
    {'x1': 4, 'x2': 'blue'} True

    And with polars:

    >>> import polars as pl
    >>> X = pl.DataFrame({
    ...     'x1': [1, 2, 3, 4],
    ...     'x2': ['blue', 'yellow', 'yellow', 'blue'],
    ...     'y': [True, False, False, True]
    ... })
    >>> y = X.get_column('y')
    >>> X = X.drop('y')

    >>> for xi, yi in stream.iter_frame(X, y):
    ...     print(xi, yi)
    {'x1': 1, 'x2': 'blue'} True
    {'x1': 2, 'x2': 'yellow'} False
    {'x1': 3, 'x2': 'yellow'} False
    {'x1': 4, 'x2': 'blue'} True

    """
    X = nw.from_native(X, eager_only=True)
    if y is not None:
        y = nw.from_native(y, eager_only=True, allow_series=True)

    kwargs["feature_names"] = X.columns
    if isinstance(y, nw.DataFrame):
        kwargs["target_names"] = y.columns

    yield from stream.iter_array(X=X.to_numpy(), y=None if y is None else y.to_numpy(), **kwargs)
