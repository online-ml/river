"""Conformal predictions. This modules contains wrappers to enable conformal predictions on any
regressor or classifier."""
from conf.base import Interval
from conf.jackknife import RegressionJackknife
from conf.gaussian import Gaussian
from conf.ACP import AdaptativeConformalPrediction
from conf.CP import ConformalPrediction

__all__ = [
    "Interval",
    "Gaussian",
    "ConformalPrediction",
    'AdaptativeConformalPrediction',
    "RegressionJackknife",
]
