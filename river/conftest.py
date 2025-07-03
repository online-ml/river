from __future__ import annotations

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
