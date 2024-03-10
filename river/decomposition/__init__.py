"""Decomposition.

"""
from __future__ import annotations

from .odmd import OnlineDMD, OnlineDMDwC
from .opca import OnlinePCA
from .osvd import OnlineSVD

__all__ = [
    "OnlineSVD",
    "OnlineDMD",
    "OnlineDMDwC",
    "OnlinePCA",
]
