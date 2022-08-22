"""Running statistics"""
from . import base
from .auto_corr import AutoCorr
from .count import Count
from .cov import Cov
from .entropy import Entropy
from .ewmean import EWMean, FastEWMean
from .ewvar import EWVar, FastEWVar
from .iqr import IQR, RollingIQR, FastIQR
from .kurtosis import Kurtosis, FastKurtosis
from .link import Link
from .lossy import LossyCount
from .mad import MAD
from .maximum import AbsMax, Max, RollingAbsMax, RollingMax
from .mean import BayesianMean, Mean
from .minimum import Min, RollingMin
from .mode import Mode, RollingMode
from .n_unique import NUnique
from .pearson import PearsonCorr
from .ptp import PeakToPeak, RollingPeakToPeak
from .quantile import Quantile, RollingQuantile, FastQuantile, FastRollingQuantile
from .sem import SEM
from .shift import Shift
from .skew import Skew, FastSkew
from .summing import Sum
from .var import Var

__all__ = [
    "base",
    "AbsMax",
    "AutoCorr",
    "BayesianMean",
    "Count",
    "Cov",
    "Entropy",
    "EWMean",
    "EWVar",
    "IQR",
    "Kurtosis",
    "Link",
    "LossyCount",
    "MAD",
    "Max",
    "Mean",
    "Min",
    "Mode",
    "NUnique",
    "PeakToPeak",
    "PearsonCorr",
    "Quantile",
    "RollingAbsMax",
    "RollingIQR",
    "RollingMax",
    "RollingMin",
    "RollingMode",
    "RollingPeakToPeak",
    "RollingQuantile",
    "SEM",
    "Shift",
    "Skew",
    "Sum",
    "Var",
    "FastQuantile",
    "FastRollingQuantile",
    "FastEWMean",
    "FastEWVar",
    "FastIQR",
    "FastKurtosis",
    "FastSkew",
]
