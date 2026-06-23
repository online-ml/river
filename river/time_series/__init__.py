"""Time series forecasting."""

from __future__ import annotations

import warnings

from . import base
from .holt_winters import HoltWinters
from .metrics import ForecastingMetric, HorizonAggMetric, HorizonMetric
from .snarimax import SNARIMAX


def iter_evaluate(*args, **kwargs):
    """Deprecated: use `evaluate.iter_evaluate` instead."""
    warnings.warn(
        "`time_series.iter_evaluate` is deprecated and will be removed in a future release; "
        "use `evaluate.iter_evaluate` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from river import evaluate

    return evaluate.iter_evaluate(*args, **kwargs)


def evaluate(*args, **kwargs):
    """Deprecated: use `evaluate.evaluate` instead."""
    warnings.warn(
        "`time_series.evaluate` is deprecated and will be removed in a future release; "
        "use `evaluate.evaluate` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from river import evaluate as _evaluate

    return _evaluate.evaluate(*args, **kwargs)


__all__ = [
    "base",
    "evaluate",
    "iter_evaluate",
    "ForecastingMetric",
    "HorizonAggMetric",
    "HorizonMetric",
    "HoltWinters",
    "SNARIMAX",
]
