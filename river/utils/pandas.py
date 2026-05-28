from __future__ import annotations

import functools
import importlib.util
import types
import typing

PANDAS_INSTALLED = importlib.util.find_spec("pandas") is not None


@functools.cache
def import_pandas() -> types.ModuleType:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "`pandas` is required for this operation. "
            'Install it with `pip install "river[pandas]"` (or `uv add "river[pandas]"`).'
        ) from exc
    return typing.cast(types.ModuleType, pd)
