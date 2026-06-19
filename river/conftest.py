from __future__ import annotations

import pathlib
import typing

import pytest

if typing.TYPE_CHECKING:
    from collections.abc import Callable

    from narwhals.stable.v2.typing import IntoDataFrame, IntoSeries

collect_ignore = []

try:
    import sklearn  # noqa: F401
except ImportError:
    collect_ignore.append("compat/test_sklearn.py")

try:
    import sqlalchemy  # noqa: F401
except ImportError:
    collect_ignore.append("stream/iter_sql.py")
    collect_ignore.append("stream/test_sql.py")

try:
    import vaex  # noqa: F401
except ImportError:
    collect_ignore.append("stream/iter_vaex.py")

try:
    import pandas  # noqa: F401
except ImportError:
    # `pandas` is an optional extra. When it is absent, skip collection of every
    # test module and doctest source that references pandas — they all need
    # `pip install "river[pandas]"` to run. Detection is text-based; we match
    # both direct pandas usage (`import pandas`, `>>> pd.`) and indirect uses
    # via sklearn's `fetch_openml`, which sklearn itself routes through pandas.
    _root = pathlib.Path(__file__).parent
    _NEEDLES = ("import pandas", ">>> pd.", "fetch_openml")
    for _path in _root.rglob("*.py"):
        try:
            _text = _path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if any(needle in _text for needle in _NEEDLES):
            collect_ignore.append(str(_path.relative_to(_root)))


class FrameBackend(typing.NamedTuple):
    """Native frame/series constructors for one dataframe library."""

    name: str
    frame: Callable[..., IntoDataFrame]
    series: Callable[..., IntoSeries]


def _pandas() -> FrameBackend:
    pd = pytest.importorskip("pandas")
    return FrameBackend(
        "pandas", pd.DataFrame, lambda values, name="y": pd.Series(values, name=name)
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
    "polars": _polars,
    "pyarrow": _pyarrow,
}


@pytest.fixture(params=list(FRAME_BACKENDS))
def frame_backend(request: pytest.FixtureRequest) -> FrameBackend:
    """Yield one `Backend` per dataframe library, skipping those that are not installed."""
    return FRAME_BACKENDS[request.param]()
