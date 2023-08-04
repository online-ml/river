"""
Concept Drift Detection.

This module contains concept drift detection methods. The purpose of a drift detector is to raise
an alarm if the data distribution changes. A good drift detector method is the one that maximizes
the true positives while keeping the number of false positives to a minimum.

"""
from __future__ import annotations

from . import binary, datasets
from .adwin import ADWIN
from .dummy import DummyDriftDetector
from .kswin import KSWIN
from .page_hinkley import PageHinkley
from .retrain import DriftRetrainingClassifier

__all__ = [
    "binary",
    "datasets",
    "ADWIN",
    "DriftRetrainingClassifier",
    "DummyDriftDetector",
    "KSWIN",
    "PageHinkley",
    "PeriodicTrigger",
]
