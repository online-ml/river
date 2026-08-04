from __future__ import annotations

import typing

import pytest

if typing.TYPE_CHECKING:
    from collections.abc import Callable

    from narwhals.stable.v2.typing import IntoDataFrame, IntoSeries


class FrameBackend(typing.NamedTuple):
    """Native frame/series constructors for one dataframe library."""

    name: str
    frame: Callable[..., IntoDataFrame]
    series: Callable[..., IntoSeries]


def _pandas() -> FrameBackend:
    pytest.importorskip("pandas")

    import pandas as pd

    return FrameBackend(
        "pandas", pd.DataFrame, lambda values, name="y": pd.Series(values, name=name)
    )


def _pandas_nullable() -> FrameBackend:
    pytest.importorskip("pandas")

    import pandas as pd

    return FrameBackend(
        "pandas[nullable]",
        lambda data: pd.DataFrame(data).convert_dtypes(dtype_backend="numpy_nullable"),
        lambda values, name="y": pd.Series(values, name=name).convert_dtypes(
            dtype_backend="numpy_nullable"
        ),
    )


def _pandas_pyarrow() -> FrameBackend:
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    import pandas as pd

    return FrameBackend(
        "pandas[pyarrow]",
        lambda data: pd.DataFrame(data).convert_dtypes(dtype_backend="pyarrow"),
        lambda values, name="y": pd.Series(values, name=name).convert_dtypes(
            dtype_backend="pyarrow"
        ),
    )


def _polars() -> FrameBackend:
    pl = pytest.importorskip("polars")
    return FrameBackend("polars", pl.DataFrame, lambda values, name="y": pl.Series(name, values))


def _pyarrow() -> FrameBackend:
    pa = pytest.importorskip("pyarrow")
    # pyarrow has no Series; its 1D analogue is a ChunkedArray, which carries no name.
    return FrameBackend("pyarrow", pa.table, lambda values, name="y": pa.chunked_array([values]))


FRAME_BACKENDS: dict[str, Callable[[], FrameBackend]] = {
    "pandas": _pandas,
    "pandas[nullable]": _pandas_nullable,
    "pandas[pyarrow]": _pandas_pyarrow,
    "polars": _polars,
    "pyarrow": _pyarrow,
}
