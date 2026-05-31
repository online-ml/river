"""Miscellaneous.

This module essentially regroups some implementations that have nowhere else to go.

"""

from __future__ import annotations

from .sdft import SDFT
from .skyline import Skyline
from .zstd_classifier import ZstdClassifier

__all__ = ["SDFT", "Skyline", "ZstdClassifier"]
