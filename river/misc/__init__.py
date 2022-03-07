"""Miscellaneous.

This module essentially regroups some implementations that have nowhere else to go.

"""
from .cov_matrix import CovMatrix
from .histogram import Histogram
from .sdft import SDFT
from .skyline import Skyline

__all__ = ["CovMatrix", "Histogram", "SDFT", "Skyline"]
