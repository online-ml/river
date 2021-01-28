"""Running statistics"""
from .auto_corr import AutoCorr
from .base import Bivariate, Univariate
from .count import Count
from .cov import Cov, RollingCov
from .entropy import Entropy
from .ewmean import EWMean
from .ewvar import EWVar
from .iqr import IQR, RollingIQR
from .kurtosis import Kurtosis
from .link import Link
from .maximum import AbsMax, Max, RollingAbsMax, RollingMax
from .mean import BayesianMean, Mean, RollingMean
from .minimum import Min, RollingMin
from .mode import Mode, RollingMode
from .n_unique import NUnique
from .pearson import PearsonCorr, RollingPearsonCorr
from .ptp import PeakToPeak, RollingPeakToPeak
from .quantile import Quantile, RollingQuantile
from .sem import SEM, RollingSEM
from .shift import Shift
from .skew import Skew
from .summing import RollingSum, Sum
from .var import RollingVar, Var

__all__ = [
    "AbsMax",
    "AutoCorr",
    "BayesianMean",
    "Bivariate",
    "Count",
    "Cov",
    "Entropy",
    "EWMean",
    "EWVar",
    "IQR",
    "Kurtosis",
    "Link",
    "Max",
    "Mean",
    "Min",
    "Mode",
    "NUnique",
    "PeakToPeak",
    "PearsonCorr",
    "Quantile",
    "RollingAbsMax",
    "RollingCov",
    "RollingIQR",
    "RollingMax",
    "RollingMean",
    "RollingMin",
    "RollingMode",
    "RollingPeakToPeak",
    "RollingPearsonCorr",
    "RollingQuantile",
    "RollingSEM",
    "RollingSum",
    "RollingVar",
    "SEM",
    "Shift",
    "Skew",
    "Sum",
    "Univariate",
    "Var",
]
