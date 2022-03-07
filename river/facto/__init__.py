"""Factorization machines."""
from .ffm import FFMClassifier, FFMRegressor
from .fm import FMClassifier, FMRegressor
from .fwfm import FwFMClassifier, FwFMRegressor
from .hofm import HOFMClassifier, HOFMRegressor

__all__ = [
    "FFMClassifier",
    "FFMRegressor",
    "FMClassifier",
    "FMRegressor",
    "FwFMClassifier",
    "FwFMRegressor",
    "HOFMClassifier",
    "HOFMRegressor",
]
