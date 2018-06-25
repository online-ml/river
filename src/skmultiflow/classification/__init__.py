"""
The :mod:`skmultiflow.classification` module includes learning method for classification.
"""

from .classifier_chains import ClassifierChain
from .multi_output_learner import MultiOutputLearner
from .naive_bayes import NaiveBayes
from .perceptron import Perceptron
from .regressor_chains import RegressorChain

__all__ = ["ClassifierChain", "MultiOutputLearner", "NaiveBayes", "Perceptron", "RegressorChain"]
