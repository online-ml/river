"""Anomaly detection.

Estimators in the `anomaly` module have a bespoke API. Each anomaly detector has a `score_one`
method instead of a `predict_one` method. This method returns an anomaly score. Normal observations
should have a low score, whereas anomalous observations should have a high score. The range of the
scores is relative to each estimator.

Anomaly detectors are usually unsupervised, in that they analyze the distribution of the features
they are shown. But River also has a notion of supervised anomaly detectors. This analyze the
distribution of a target variable, and optionally the distribution of the features as well. These
anomaly detectors are univariate. They are useful for detecting labelling anomalies, which can be
detrimental if they learned by a model.

An unsupervised anomaly detector can be combined with a thresholding method to convert its anomaly
scores into booleans, thus turning it into a binary classifier of sorts.

"""
from . import base
from .hst import HalfSpaceTrees
from .svm import OneClassSVM
from .threshold import ConstantThresholder, QuantileThresholder
from .gaussian import Gaussian

__all__ = [
    "base",
    "ConstantThresholder",
    "HalfSpaceTrees",
    "QuantileThresholder",
    "OneClassSVM",
    "Gaussian",
]
