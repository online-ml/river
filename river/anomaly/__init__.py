"""Anomaly detection.

Estimators in the `anomaly` module have a bespoke API. Each anomaly detector has a `score_one`
method instead of a `predict_one` method. This method returns an anomaly score. A high score
indicates an anomaly, whereas a low score is indicative of a normal observation. The range of the
scores is relative to each estimator.

An anomaly detector can be combined with a thresholding method to turn its anomaly scores into 0s
and 1s.

"""
from .base import AnomalyDetector
from .hst import HalfSpaceTrees
from .svm import OneClassSVM
from .threshold import ConstantThresholder, QuantileThresholder

__all__ = [
    "AnomalyDetector",
    "ConstantThresholder",
    "HalfSpaceTrees",
    "QuantileThresholder",
    "OneClassSVM",
]
