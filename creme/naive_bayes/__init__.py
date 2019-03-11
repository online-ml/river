"""
Naive Bayes algorithms.
"""
from .gaussian import GaussianNB
from .multinomial import MultinomialNB


__all__ = ['GaussianNB', 'MultinomialNB']
