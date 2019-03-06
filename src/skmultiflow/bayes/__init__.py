"""
The :mod:`skmultiflow.bayes` module includes Bayes learning methods.
"""

from .utils import do_naive_bayes_prediction
from .naive_bayes import NaiveBayes

__all__ = ["do_naive_bayes_prediction", "NaiveBayes"]
