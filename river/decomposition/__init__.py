"""Decomposition.

"""
from __future__ import annotations

from .odmd import OnlineDMD, OnlineDMDwC
from .opca import OnlinePCA
from .osvd import OnlineSVD, OnlineSVDZhang

__all__ = [
    "OnlineSVD",
    "OnlineSVDZhang",
    "OnlineDMD",
    "OnlineDMDwC",
    "OnlinePCA",
]
