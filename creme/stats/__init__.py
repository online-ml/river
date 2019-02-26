"""
Module for computing running statistics
"""
from .count import Count
from .ewmean import EWMean
from .kurtosis import Kurtosis
from .max import Max
from .mean import Mean
from .min import Min
from .mode import Mode
from .n_unique import NUnique
from .ptp import PeakToPeak
from .skew import Skew
from .sum import Sum
from .variance import Variance
from .quantile import Quantile
from .quantile import RollingQuantile
from .categorical_count import CategoricalCount
from .entropy import Entropy
from .sorted_window import _SortedWindow


__all__ = [
    'Count',
    'EWMean',
    'Kurtosis',
    'Max',
    'Mean',
    'Min',
    'Mode',
    'NUnique',
    'PeakToPeak',
    'Quantile',
    'Skew',
    'Sum',
    'Variance',
    'Quantile',
    'RollingQuantile',
    'CategoricalCount',
    'Entropy',
    '_SortedWindow'
]
