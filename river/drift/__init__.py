"""
Concept Drift Detection.

This module contains concept drift detection methods. The purpose of a drift detector is to raise
an alarm if the data distribution changes. A good drift detector method is the one that maximizes
the true positives while keeping the number of false positives to a minimum.

"""

from .adwin import ADWIN
from .ddm import DDM
from .eddm import EDDM
from .page_hinkley import PageHinkley
from .hddm_a import HDDM_A
from .hddm_w import HDDM_W
from .kswin import KSWIN

__all__ = ["ADWIN", "DDM", "EDDM", "PageHinkley", "HDDM_A", "HDDM_W", "KSWIN"]
