"""Miscellaneous.

This module essentially regroups some implementations that have nowhere else to go.

"""
from .cov_matrix import CovMatrix
from .inv_cov_matrix import InvCovMatrix
from .histogram import Histogram
from .sdft import SDFT
from .skyline import Skyline

__all__ = ["CovMatrix", "InvCovMatrix", "Histogram", "SDFT", "Skyline"]
