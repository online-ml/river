"""Anomaly detection.

The estimators in ``anomaly`` are slightly different than the rest of the estimators. Instead of a
``predict_one`` method, each anomaly detector has a ``score_one`` method which returns an anomaly
score for a given set of features. Anomalies will have high scores whereas normal observations will
have low scores. The range of the scores depends on the estimator.

"""
from .hst import HalfSpaceTrees


__all__ = ['HalfSpaceTrees']
