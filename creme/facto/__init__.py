"""Factorization Machines models."""
from .ffm import FFMClassifier
from .ffm import FFMRegressor
from .fm import FMClassifier
from .fm import FMRegressor
from .fwfm import FWFMClassifier
from .fwfm import FWFMRegressor
from .hofm import HOFMClassifier
from .hofm import HOFMRegressor


__all__ = [
    'FFMClassifier',
    'FFMRegressor',
    'FMClassifier',
    'FMRegressor',
    'FWFMClassifier',
    'FWFMRegressor',
    'HOFMClassifier',
    'HOFMRegressor',
]
