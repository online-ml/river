from __future__ import annotations

import os
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

# Text-based detection of pandas usage, matching both direct pandas usage
# (`import pandas`, `>>> pd.`) and indirect uses via sklearn's `fetch_openml`,
# which sklearn itself routes through pandas.
_ROOT = pathlib.Path(__file__).parent
_PANDAS_NEEDLES = ("import pandas", ">>> pd.", "fetch_openml")


def _uses_pandas(path: pathlib.Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return any(needle in text for needle in _PANDAS_NEEDLES)


if os.environ.get("RIVER_PANDAS_ONLY") == "1":
    # Companion CI job: pandas IS installed and we run *only* the pandas subset,
    # so the main (no-pandas) job doesn't have to re-run everything else. Ignore
    # every source/test module that does not touch pandas.
    for _path in _ROOT.rglob("*.py"):
        if _path.name != "conftest.py" and not _uses_pandas(_path):
            collect_ignore.append(str(_path.relative_to(_ROOT)))
else:
    try:
        import pandas  # noqa: F401
    except ImportError:
        # `pandas` is an optional extra. When it is absent, skip collection of
        # every test module and doctest source that references pandas — they all
        # need `pip install "river[pandas]"` to run.
        for _path in _ROOT.rglob("*.py"):
            if _uses_pandas(_path):
                collect_ignore.append(str(_path.relative_to(_ROOT)))


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


@pytest.fixture(params=list(FRAME_BACKENDS))
def frame_backend(request: pytest.FixtureRequest) -> FrameBackend:
    """Yield one `Backend` per dataframe library, skipping those that are not installed."""
    return FRAME_BACKENDS[request.param]()
