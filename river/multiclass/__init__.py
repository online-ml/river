"""Multi-class classification."""
from __future__ import annotations

from .occ import OutputCodeClassifier
from .ovo import OneVsOneClassifier
from .ovr import OneVsRestClassifier

__all__ = ["OutputCodeClassifier", "OneVsOneClassifier", "OneVsRestClassifier"]
