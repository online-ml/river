"""
Module for computing running statistics
"""
from .auto_corr import AutoCorrelation
from .base import Bivariate
from .base import Univariate
from .count import Count
from .covariance import Covariance
from .entropy import Entropy
from .ewmean import EWMean
from .ewvar import EWVar
from .kurtosis import Kurtosis
from .max import Max
from .max import RollingMax
from .mean import BayesianMean
from .mean import Mean
from .mean import RollingMean
from .min import Min
from .min import RollingMin
from .mode import Mode
from .mode import RollingMode
from .n_unique import NUnique
from .pearson import PearsonCorrelation
from .ptp import PeakToPeak
from .ptp import RollingPeakToPeak
from .quantile import Quantile
from .quantile import RollingQuantile
from .sem import SEM
from .sem import RollingSEM
from .skew import Skew
from .sum import Sum
from .sum import RollingSum
from .var import Var
from .var import RollingVar


__all__ = [
    'AutoCorrelation',
    'BayesianMean',
    'Bivariate',
    'Count',
    'Covariance',
    'Entropy',
    'EWMean',
    'EWVar',
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
    'RollingVar',
    'SEM',
    'Skew',
    'Sum',
    'Univariate',
    'Var'
]
