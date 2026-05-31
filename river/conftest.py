from __future__ import annotations

import pathlib

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
