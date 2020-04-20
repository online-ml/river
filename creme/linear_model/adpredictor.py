import collections
import math 
import numbers

import numpy as np 
import scipy as sp 
from .. import base 
from .. import optim
from .. import proba



__all__ = [
    'AdPredictor'
]


class NormalPrior(optim.initializers.Initializer):
    """Initializer which simulate a normal distribution based on the prior probability on weight.
    The mean of the normal distribution (mu) is calculated through the predictive posterior closed form.
    
    Parameters:
        prior_probability (float): prior probability on the feature weights prior_probability = P(y=1 | x, weights)
        beta (float): scale of the inverse link function (cumulative distribution function of standard normal distribution)
        n_features (int): number of features
    """
    def __init__(self, prior_probability = None, beta, n_features):
        self.prior_probability = prior_probability
        self.beta = beta
        self.n_features = n_features
        self.mu = O if prior_probability is None else sp.stats.norm.ppf(prior_probability) * (beta ** 2 + n_features)
        self.sigma = 1.

    def __str__(self):
        return f'Unbiased normal prior on weights 𝒩(μ={self.mu:.3f}, σ={self.sigma:.3f})' if prior_probability is None else f'Biased normal prior on weights 𝒩(μ={self.mu:.3f}, σ={self.sigma:.3f})'

class BOPR:
    """Bayesian Online Linear Regression Model.
    """
    pass


class AdPredictor(BOPR, base.Regressor):
    """AdPredictor.
    """

    pass