"""Miscellaneous.

This module essentially regroups some implementations that have nowhere else to go.

"""
from .cms import CountMin
from .histogram import Histogram
from .sdft import SDFT
from .skyline import Skyline

__all__ = ["CountMin", "Histogram", "SDFT", "Skyline"]
