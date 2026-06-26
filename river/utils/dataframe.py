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
import numpy as np

if typing.TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from typing import Any

    from narwhals.stable.v2.typing import (
        IntoDataFrame,
        IntoDataFrameT,
        IntoSeries,
        IntoSeriesT,
    )
    from numpy.typing import NDArray
    from scipy import sparse

__all__ = [
    "into_frame",
    "into_series",
    "sparse_to_native_frame",
    "to_native_frame",
    "to_native_series",
    "to_numpy",
]


def to_numpy(frame: nw.DataFrame[Any]) -> NDArray[np.float64]:
    """Extract a `float64` numpy matrix from a narwhals dataframe for the numpy compute core.

    A pandas frame backed by pyarrow (`ArrowDtype`) columns returns an ``object`` array from
    ``.to_numpy()``, which breaks downstream ufuncs (e.g. ``np.exp`` raises *"loop of ufunc does
    not support argument 0 of type float"*). Coercing to ``float64`` at the boundary keeps the
    core backend-agnostic; ``np.asarray`` is a no-op when the frame already yields ``float64``.
    """
    return np.asarray(frame.to_numpy(), dtype=np.float64)


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
    values: NDArray[Any] | Sequence[Any], *, name: str | None, like: nw.DataFrame[IntoDataFrame]
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


def to_native_frame(
    data: Mapping[Any, NDArray[Any] | Sequence[Any]], *, like: nw.DataFrame[Any]
) -> IntoDataFrame:
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


def sparse_to_native_frame(
    matrix: sparse.spmatrix, *, columns: Iterable[Any], like: nw.DataFrame[Any] | nw.Series[Any]
) -> IntoDataFrame:
    """Build a native frame from a scipy sparse matrix, matching the backend of `like`.

    Term-document matrices (e.g. from `feature_extraction.BagOfWords`/`TFIDF`) are very
    sparse, but only pandas exposes a per-column sparse dtype: Arrow's sparse types are
    standalone tensors rather than table columns, and polars has no sparse encoding. So
    pandas-like backends keep the memory-efficient sparse representation, while every other
    backend is densified.

    Parameters
    ----------
    matrix
        A `scipy.sparse` matrix (typically CSR) produced by the numpy core, shaped
        `(n_documents, n_terms)`.
    columns
        The term labels, in column order. May contain n-gram tuples; non-pandas backends
        require string column names, so labels are stringified for them.
    like
        The narwhals frame or series the call received as input. Its backend determines the
        return type, and its index (pandas only) is carried over to the result.
    """
    impl = like.implementation
    if impl.is_pandas_like():
        # Use the input's own pandas namespace so pandas[nullable]/pandas[pyarrow] are honoured.
        ns = nw.get_native_namespace(like)
        frame = ns.DataFrame.sparse.from_spmatrix(matrix, columns=list(columns))
    else:
        dense = matrix.toarray()
        data = {str(col): dense[:, j] for j, col in enumerate(columns)}
        frame = nw.from_dict(data, backend=impl).to_native()
    # Carry over the pandas index; no-op for non-pandas backends (maybe_get_index -> None).
    if (index := nw.maybe_get_index(like)) is not None:
        frame.index = index
    return typing.cast("IntoDataFrame", frame)
