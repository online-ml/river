from __future__ import annotations

import typing
import warnings

import narwhals.stable.v2 as nw

from river import base, stream

if typing.TYPE_CHECKING:
    import polars as pl


def iter_polars(
    X: pl.DataFrame, y: pl.Series | pl.DataFrame | None = None, **kwargs: typing.Any
) -> base.typing.Stream:
    """Iterates over the rows of a `polars.DataFrame`.

    .. deprecated::
        Use `stream.iter_frame` instead, which works with any eager dataframe (pandas, polars,
        PyArrow, ...). `stream.iter_polars` will be removed in a future release.

    Parameters
    ----------
    X
        A dataframe of features.
    y
        A series or a dataframe with one column per target.
    kwargs
        Extra keyword arguments are passed to the underlying call to `stream.iter_frame`
        (e.g. `shuffle` and `seed`).

    Examples
    --------

    >>> import polars as pl
    >>> from river import stream

    >>> X = pl.DataFrame({
    ...     'x1': [1, 2, 3, 4],
    ...     'x2': ['blue', 'yellow', 'yellow', 'blue'],
    ...     'y': [True, False, False, True]
    ... })
    >>> y = X.get_column('y')
    >>> X = X.drop('y')

    >>> for xi, yi in stream.iter_polars(X, y):
    ...     print(xi, yi)
    {'x1': 1, 'x2': 'blue'} True
    {'x1': 2, 'x2': 'yellow'} False
    {'x1': 3, 'x2': 'yellow'} False
    {'x1': 4, 'x2': 'blue'} True

    """
    warnings.warn(
        "`stream.iter_polars` is deprecated; use `stream.iter_frame` instead. "
        "It will be removed in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )

    if not nw.dependencies.is_polars_dataframe(X):
        raise TypeError(f"Expected a polars DataFrame, got {type(X)}")
    if y is not None and not (
        nw.dependencies.is_polars_dataframe(y) or nw.dependencies.is_polars_series(y)
    ):
        raise TypeError(f"Expected a polars DataFrame or Series, got {type(y)}")

    yield from stream.iter_frame(X, y, **kwargs)
