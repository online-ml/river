"""This module implements forest-based classifiers and regressors."""


from .adaptive_random_forest import ARFClassifier, ARFRegressor
from .aggregated_mondrian_forest import AMFClassifier
from .online_extra_trees import OXTRegressor

__all__ = [
    "ARFClassifier",
    "ARFRegressor",
    "AMFClassifier",
    "OXTRegressor",
]
