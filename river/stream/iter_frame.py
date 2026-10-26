from __future__ import annotations

import random
import typing

import narwhals.stable.v2 as nw

from river import base

if typing.TYPE_CHECKING:
    from narwhals.stable.v2.typing import IntoDataFrame, IntoFrame, IntoSeries


def iter_frame(
    X: IntoFrame,
    y: IntoSeries | IntoDataFrame | None = None,
    *,
    shuffle: bool = False,
    seed: int | None = None,
) -> base.typing.Stream:
    """Iterates over the rows of a dataframe.

    This is a dataframe-agnostic iterator: it works with any eager dataframe supported by
    [Narwhals](https://narwhals-dev.github.io/narwhals/) (pandas, polars, PyArrow, Modin, cuDF,
    ...). It supersedes `stream.iter_pandas` and `stream.iter_polars`.

    Rows are read directly from the dataframe via Narwhals, so each cell keeps its native
    per-column type (no conversion to a single `numpy` array, which would otherwise coerce a
    mixed-type frame to a common dtype).

    Note that vaex is *not* supported here: Narwhals only exposes it through the dataframe
    interchange protocol, which cannot iterate rows without materializing the whole frame. Use
    `stream.iter_vaex` instead, which streams a vaex dataframe lazily.

    Parameters
    ----------
    X
        A dataframe of features. Any eager dataframe supported by Narwhals will work.
    y
        A series, or a dataframe with one column per target.
    shuffle
        Whether to shuffle the rows before iterating over them. This materializes the whole
        stream in memory, as the order can only be permuted once every row is known.
    seed
        Random seed used for shuffling. Only used when `shuffle` is `True`.

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

    With polars:

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

    And with PyArrow:

    >>> import pyarrow as pa
    >>> X = pa.table({
    ...     'x1': [1, 2, 3, 4],
    ...     'x2': ['blue', 'yellow', 'yellow', 'blue'],
    ...     'y': [True, False, False, True]
    ... })
    >>> y = X.column('y')
    >>> X = X.drop(['y'])

    >>> for xi, yi in stream.iter_frame(X, y):
    ...     print(xi, yi)
    {'x1': 1, 'x2': 'blue'} True
    {'x1': 2, 'x2': 'yellow'} False
    {'x1': 3, 'x2': 'yellow'} False
    {'x1': 4, 'x2': 'blue'} True

    """
    # Accept any frame so we can raise an explicit error for the lazy case, rather than letting
    # Narwhals reject it with a generic message.
    frame = nw.from_native(X, allow_series=False)
    if isinstance(frame, nw.LazyFrame):
        raise TypeError(
            f"`stream.iter_frame` only supports eager dataframes, but got a lazy frame "
            f"({type(X).__name__!r}). Iterating rows would force a full materialization, and most "
            f"lazy backends do not guarantee row order. Collect it first (e.g. via `.collect()`) "
            f"and pass the resulting eager frame."
        )

    # Narwhals types row keys as `str`; River's `FeatureName` is the broader `Hashable`, and
    # `dict` keys are invariant, hence the cast.
    x_iter = typing.cast(
        "typing.Iterator[dict[base.typing.FeatureName, typing.Any]]",
        frame.iter_rows(named=True),
    )

    rows: typing.Iterable[tuple[dict[base.typing.FeatureName, typing.Any], typing.Any]]
    if y is None:
        rows = ((x_row, None) for x_row in x_iter)
    else:
        target = nw.from_native(y, eager_only=True, allow_series=True)
        y_iter = target.iter_rows(named=True) if isinstance(target, nw.DataFrame) else target
        rows = zip(x_iter, y_iter)

    if shuffle:
        # Permuting the order requires knowing every row, hence the materialization.
        rows = list(rows)
        random.Random(seed).shuffle(rows)

    yield from rows
