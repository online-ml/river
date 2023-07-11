"""This module implements forest-based classifiers and regressors."""
from __future__ import annotations

from .adaptive_random_forest import ARFClassifier, ARFRegressor
from .aggregated_mondrian_forest import AMFClassifier, AMFRegressor
from .online_extra_trees import OXTRegressor

__all__ = [
    "ARFClassifier",
    "ARFRegressor",
    "AMFClassifier",
    "AMFRegressor",
    "OXTRegressor",
]
