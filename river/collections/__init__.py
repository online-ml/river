"""Data collections for sequential data.

This module has summary and sketch structures that operate with
constrained amounts of memory and processing time.

"""
from .counter import Counter
from .histogram import Histogram
from .heavy_hitters import HeavyHitters
from .sdft import SDFT
from .skyline import Skyline

__all__ = ["Counter", "Histogram", "HeavyHitters", "SDFT", "Skyline"]
