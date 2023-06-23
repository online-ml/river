"""Running statistics"""
from __future__ import annotations

from . import base
from .auto_corr import AutoCorr
from .count import Count
from .cov import Cov
from .entropy import Entropy
from .ewmean import EWMean
from .ewvar import EWVar
from .iqr import IQR, RollingIQR
from .kurtosis import Kurtosis
from .link import Link
from .mad import MAD
from .maximum import AbsMax, Max, RollingAbsMax, RollingMax
from .mean import BayesianMean, Mean
from .minimum import Min, RollingMin
from .mode import Mode, RollingMode
from .n_unique import NUnique
from .pearson import PearsonCorr
from .ptp import PeakToPeak, RollingPeakToPeak
from .quantile import Quantile, RollingQuantile
from .sem import SEM
from .shift import Shift
from .skew import Skew
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
]
