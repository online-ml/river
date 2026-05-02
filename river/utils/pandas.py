from __future__ import annotations

import functools
import importlib.util
import typing

if typing.TYPE_CHECKING:
    import pandas as pd


PANDAS_INSTALLED = importlib.util.find_spec("pandas") is not None


def import_pandas() -> pd:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise ImportError("`pandas` is required for this operation.") from exc
    return pd


def requires_pandas(method):
    @functools.wraps(method)
    def wrapper(*args, **kwargs):
        import_pandas()
        return method(*args, **kwargs)

    return wrapper
