"""Miscellaneous.

This module essentially regroups some implementations that have nowhere else to go.

"""
from .count_min import CountMin
from .histogram import Histogram
from .lossy import LossyCount
from .sdft import SDFT
from .skyline import Skyline

__all__ = ["CountMin", "Histogram", "LossyCount", "SDFT", "Skyline"]
