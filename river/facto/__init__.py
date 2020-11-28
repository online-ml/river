"""Factorization machines."""
from .ffm import FFMClassifier
from .ffm import FFMRegressor
from .fm import FMClassifier
from .fm import FMRegressor
from .fwfm import FwFMClassifier
from .fwfm import FwFMRegressor
from .hofm import HOFMClassifier
from .hofm import HOFMRegressor


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
