"""
Module for computing running statistics
"""
from .base import Bivariate
from .base import Univariate
from .count import Count
from .covariance import Covariance
from .entropy import Entropy
from .ewmean import EWMean
from .kurtosis import Kurtosis
from .max import Max
from .mean import BayesianMean
from .mean import Mean
from .min import Min
from .mode import Mode
from .n_unique import NUnique
from .pearson import PearsonCorrelation
from .ptp import PeakToPeak
from .quantile import Quantile
from .rolling_max import RollingMax
from .rolling_mean import RollingMean
from .rolling_min import RollingMin
from .rolling_mode import RollingMode
from .rolling_ptp import RollingPeakToPeak
from .rolling_quantile import RollingQuantile
from .rolling_sem import RollingSEM
from .rolling_sum import RollingSum
from .rolling_variance import RollingVariance
from .sem import SEM
from .skew import Skew
from .sum import Sum
from .variance import Variance


__all__ = [
    'BayesianMean',
    'Bivariate',
    'Count',
    'Covariance',
    'Entropy',
    'EWMean',
    'Kurtosis',
    'Max',
    'Mean',
    'Min',
    'Mode',
    'NUnique',
    'PeakToPeak',
    'PearsonCorrelation',
    'Quantile',
    'RollingMean',
    'RollingQuantile',
    'RollingMax',
    'RollingMin',
    'RollingMode',
    'RollingPeakToPeak',
    'RollingSEM',
    'RollingSum',
    'RollingVariance',
    'SEM',
    'Skew',
    'Sum',
    'Univariate',
    'Variance'
]
