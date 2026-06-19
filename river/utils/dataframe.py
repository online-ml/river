"""Dataframe-agnostic boundary helpers for mini-batch methods.

River's ``*_many`` methods compute in numpy/scipy but historically took and returned
``pandas`` objects. These helpers move the pandas dependency to the boundary via
[narwhals](https://github.com/narwhals-dev/narwhals): the input is wrapped on entry, the
numpy core is left untouched, and the output is rebuilt with the *same* backend (and pandas
index) as the input. This lets any narwhals-supported eager backend (pandas, polars,
pyarrow, ...) flow through unchanged.

See `river/utils/pandas.py` for the pandas-optional import helper.
"""

from __future__ import annotations

import typing

import narwhals.stable.v2 as nw

if typing.TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any

    from narwhals.stable.v2.typing import (
        IntoDataFrame,
        IntoDataFrameT,
        IntoSeries,
        IntoSeriesT,
    )

__all__ = ["into_frame", "into_series", "to_native_frame", "to_native_series"]


def into_frame(X: IntoDataFrameT) -> nw.DataFrame[IntoDataFrameT]:
    """Wrap a native eager dataframe in a narwhals `DataFrame`.

    The native frame type flows through the `IntoDataFrameT` type variable, so callers keep a
    precisely-typed handle (e.g. `nw.DataFrame[pd.DataFrame]`) instead of a backend union.
    """
    return nw.from_native(X, eager_only=True)


def into_series(y: IntoSeriesT) -> nw.Series[IntoSeriesT]:
    """Wrap a native eager series in a narwhals `Series`."""
    return nw.from_native(y, series_only=True)


def to_native_series(
    values: Sequence[Any], *, name: str | None, like: nw.DataFrame[IntoDataFrame]
) -> IntoSeries:
    """Build a native series matching the backend (and pandas index) of `like`.

    Parameters
    ----------
    values
        The values of the series, typically a numpy array produced by the numpy core.
    name
        The name to give the series.
    like
        The narwhals dataframe the call received as input. Its backend determines the
        return type, and its index (pandas only) is carried over to the result.
    """
    # TODO(FBruzzesi): narwhals' stub types `name` as `str`, but every backend accepts `None` at runtime.
    series = nw.new_series(name=name, values=values, backend=nw.get_native_namespace(like))  # type: ignore[arg-type]
    native = series.to_native()
    # Carry over the pandas index; no-op for non-pandas backends (maybe_get_index -> None).
    if (index := nw.maybe_get_index(like)) is not None:
        native.index = index
    return typing.cast("IntoSeries", native)


def to_native_frame(data: Mapping[Any, Sequence[Any]], *, like: nw.DataFrame[Any]) -> IntoDataFrame:
    """Build a native dataframe matching the backend (and pandas index) of `like`.

    Pandas-like backends accept arbitrary column labels (e.g. the booleans `False`/`True`
    used by binary classifiers), so they are preserved as-is. Every other backend requires
    string column names, so labels are stringified for them.

    Parameters
    ----------
    data
        A mapping from column label to column values (numpy arrays from the numpy core).
    like
        The narwhals dataframe the call received as input. Its backend determines the
        return type, and its index (pandas only) is carried over to the result.
    """
    impl = like.implementation
    if not impl.is_pandas_like():
        data = {str(key): value for key, value in data.items()}
    frame = nw.from_dict(data, backend=impl).to_native()
    # Carry over the pandas index; no-op for non-pandas backends (maybe_get_index -> None).
    if (index := nw.maybe_get_index(like)) is not None:
        frame.index = index
    return typing.cast("IntoDataFrame", frame)
