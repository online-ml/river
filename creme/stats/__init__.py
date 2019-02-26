"""
Module for computing running statistics
"""
from .count import Count
from .entropy import Entropy
from .ewmean import EWMean
from .kurtosis import Kurtosis
from .max import Max
from .mean import Mean
from .min import Min
from .mode import Mode
from .n_unique import NUnique
from .ptp import PeakToPeak
from .quantile import Quantile
from .quantile import RollingQuantile
from .skew import Skew
from .sum import Sum
from .variance import Variance


__all__ = [
    'Count',
    'Entropy',
    'EWMean',
    'Kurtosis',
    'Max',
    'Mean',
    'Min',
    'Mode',
    'NUnique',
    'PeakToPeak',
    'Quantile',
    'RollingQuantile',
    'Skew',
    'Sum',
    'Variance'
]
