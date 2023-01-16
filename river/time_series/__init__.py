"""Time series forecasting."""
from . import base
from .evaluates import evaluate, get_iter_evaluate, iter_evaluate, _iter_with_horizon
# renaming module to evaluates to be able to import more than 1 function
from .holt_winters import HoltWinters
from .metrics import ForecastingMetric, HorizonAggMetric, HorizonMetric
from .snarimax import SNARIMAX
from .intervals import ForecastingInterval, HorizonInterval
from .AdHoeffTree_horizon import AdaptHoeffdingHorizon

__all__ = [
    "base",
    "_iter_with_horizon",
    "iter_evaluate",
    "get_iter_evaluate",
    "evaluate",
    "ForecastingInterval",
    "HorizonInterval",
    "ForecastingMetric",
    "HorizonAggMetric",
    "HorizonMetric",
    "AdaptHoeffdingHorizon",
    "HoltWinters",
    "SNARIMAX",
]
