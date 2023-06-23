"""Anomaly detection.

Estimators in the `anomaly` module have a bespoke API. Each anomaly detector has a `score_one`
method instead of a `predict_one` method. This method returns an anomaly score. Normal observations
should have a low score, whereas anomalous observations should have a high score. The range of the
scores is relative to each estimator.

Anomaly detectors are usually unsupervised, in that they analyze the distribution of the features
they are shown. But River also has a notion of supervised anomaly detectors. These analyze the
distribution of a target variable, and optionally include the distribution of the features as well. They are useful for detecting labelling anomalies, which can be detrimental if they learned by a
model.

"""
from __future__ import annotations

from . import base
from .filter import QuantileFilter, ThresholdFilter
from .gaussian import GaussianScorer
from .hst import HalfSpaceTrees
from .svm import OneClassSVM

__all__ = [
    "base",
    "AnomalyDetector",
    "GaussianScorer",
    "HalfSpaceTrees",
    "OneClassSVM",
    "QuantileFilter",
    "ThresholdFilter",
]
