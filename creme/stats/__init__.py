"""
Module for computing running statistics
"""
from .count import Count
from .kurtosis import Kurtosis
from .max import Max
from .mean import Mean
from .mean import SmoothMean
from .min import Min
from .n_unique import NUnique
from .p2p import PeakToPeak
from .skew import Skew
from .sum import Sum
from .variance import Variance

__all__ = [
    'Count',
    'Kurtosis',
    'Max',
    'Mean',
    'Min',
    'NUnique',
    'PeakToPeak',
    'Skew',
    'SmoothMean',
    'Sum',
    'Variance'
]
