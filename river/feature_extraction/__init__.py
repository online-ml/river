"""Feature extraction.

This module can be used to extract information from raw features. This includes encoding
categorical data as well as looking at interactions between existing features. This differs from
the `processing` module in that the latter's purpose is rather to clean the data so that it may
be processed by a particular machine learning algorithm.

"""
from .agg import Agg, TargetAgg
from .kernel_approx import RBFSampler
from .poly import PolynomialExtender
from .vectorize import TFIDF, BagOfWords

__all__ = [
    "Agg",
    "BagOfWords",
    "PolynomialExtender",
    "RBFSampler",
    "TargetAgg",
    "TFIDF",
]
