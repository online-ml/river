from __future__ import annotations

import os
import pathlib
import typing

import pytest

from tests.frames import FRAME_BACKENDS, FrameBackend

collect_ignore = []

try:
    import sklearn  # noqa: F401
except ImportError:
    collect_ignore.append("tests/compat/test_sklearn.py")

try:
    import sqlalchemy  # noqa: F401
except ImportError:
    collect_ignore.append("river/stream/iter_sql.py")
    collect_ignore.append("tests/stream/test_sql.py")

try:
    import vaex  # noqa: F401
except ImportError:
    collect_ignore.append("river/stream/iter_vaex.py")

# Text-based detection of pandas usage, matching both direct pandas usage
# (`import pandas`, `>>> pd.`) and indirect uses via sklearn's `fetch_openml`,
# which sklearn itself routes through pandas.
_ROOT = pathlib.Path(__file__).parent
# The library (doctest sources) and the test suite are separate trees since the
# tests were moved out of the package; both need the same pandas filtering.
_TREES = (_ROOT / "river", _ROOT / "tests")
_PANDAS_NEEDLES = ("import pandas", ">>> pd.", "fetch_openml")


def _iter_modules() -> typing.Iterator[pathlib.Path]:
    for tree in _TREES:
        yield from tree.rglob("*.py")


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
    for _path in _iter_modules():
        if _path.name != "conftest.py" and not _uses_pandas(_path):
            collect_ignore.append(str(_path.relative_to(_ROOT)))
else:
    try:
        import pandas  # noqa: F401
    except ImportError:
        # `pandas` is an optional extra. When it is absent, skip collection of
        # every test module and doctest source that references pandas — they all
        # need `pip install "river[pandas]"` to run.
        for _path in _iter_modules():
            if _uses_pandas(_path):
                collect_ignore.append(str(_path.relative_to(_ROOT)))


@pytest.fixture(params=list(FRAME_BACKENDS))
def frame_backend(request: pytest.FixtureRequest) -> FrameBackend:
    """Yield one `Backend` per dataframe library, skipping those that are not installed."""
    return FRAME_BACKENDS[request.param]()
