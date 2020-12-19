"""Anomaly detection.

The estimators in the `anomaly` module have a slightly different API. Instead of a `predict_one`
method, each anomaly detector has a `score_one`. The latter returns an anomaly score for a given
set of features. High scores indicate anomalies, whereas low scores indicate normal observations.
Note that the range of the scores is relative to each estimator.

"""
from .hst import HalfSpaceTrees


__all__ = ["HalfSpaceTrees"]
