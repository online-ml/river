from __future__ import annotations

import functools
import importlib.util
import types
import typing

PANDAS_INSTALLED = importlib.util.find_spec("pandas") is not None

F = typing.TypeVar("F", bound=typing.Callable[..., typing.Any])


def import_pandas() -> types.ModuleType:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise ImportError("`pandas` is required for this operation.") from exc
    return typing.cast(types.ModuleType, pd)


def requires_pandas(method: F) -> F:
    @functools.wraps(method)
    def wrapper(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        import_pandas()
        return method(*args, **kwargs)

    return typing.cast(F, wrapper)
