"""Running statistics"""
from .auto_corr import AutoCorr
from .base import Bivariate
from .base import Univariate
from .count import Count
from .cov import Cov
from .cov import RollingCov
from .entropy import Entropy
from .ewmean import EWMean
from .ewvar import EWVar
from .iqr import IQR
from .iqr import RollingIQR
from .link import Link
from .kurtosis import Kurtosis
from .maximum import Max
from .maximum import RollingMax
from .maximum import AbsMax
from .maximum import RollingAbsMax
from .mean import BayesianMean
from .mean import Mean
from .mean import RollingMean
from .minimum import Min
from .minimum import RollingMin
from .mode import Mode
from .mode import RollingMode
from .n_unique import NUnique
from .pearson import PearsonCorr
from .pearson import RollingPearsonCorr
from .ptp import PeakToPeak
from .ptp import RollingPeakToPeak
from .quantile import Quantile
from .quantile import RollingQuantile
from .shift import Shift
from .sem import SEM
from .sem import RollingSEM
from .skew import Skew
from .summing import Sum
from .summing import RollingSum
from .var import Var
from .var import RollingVar


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
