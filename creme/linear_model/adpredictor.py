import collections
import numpy as np 
from scipy import stats
from .. import base 
from .. import optim
from .. import utils



__all__ = [
    'AdPredictor'
]


class NormalPrior(optim.initializers.Initializer):
    """Initializer which simulate a normal prior distribution on weight.
    The mean of the normal distribution (mean) is calculated through the predictive posterior closed form.

    Parameters:
        prior_probability (float): prior probability on the feature weights prior_probability = P(y=1 | x, weights)
        beta (float): scale of the inverse link function (cumulative distribution function of standard normal distribution)
        n_features (int): number of features
    """
    def __init__(self, beta, n_features, prior_probability=None):
        self.prior_probability = prior_probability
        self.beta = beta
        self.n_features = n_features
        self.mean = 0 if prior_probability is None else stats.norm.ppf(prior_probability) * (beta ** 2 + n_features)
        self.variance = 1.0

    def __str__(self):
        return f'Unbiased normal prior on weights 𝒩(μ={self.mean:.3f}, σ2={self.variance:.3f})' if self.prior_probability is None else f'Biased normal prior on weights 𝒩(μ={self.mean:.3f}, σ2={self.variance:.3f})'

    def __call__(self):
        return self


class BOPR:
    """Bayesian Online Probit Regression Model.
    """
    def __init__(self, initializer, surprise, epsilon):
        self.surprise = surprise
        self.epsilon = epsilon
        self.beta = initializer.beta
        self.prior_probability = initializer.prior_probability
        self.initializer = initializer
        self.weights = collections.defaultdict(initializer)

    def _active_mean_variance(self, x):

        means = [self.weights[i].mean for i, xi in x.items()]
        variances = [self.weights[i].variance for i, xi in x.items()]

        return sum(means), sum(variances) + self.beta ** 2

    def _gaussian_correction(self, t):

        t = utils.math.clamp(t, minimum=-self.surprise, maximum=self.surprise)
        v = stats.norm.pdf(t)/stats.norm.cdf(t)
        w = v * (v + t)
        return (v, w)

    def _apply_dynamics(self, weight):
        prior = self.initializer
        adjusted_variance = weight.variance * prior.variance / ((1.0 - self.epsilon) * prior.variance + self.epsilon * weight.variance)
        adjusted_mean = adjusted_variance * ((1.0 - self.epsilon) * weight.mean / weight.variance + self.epsilon * prior.mean / prior.variance)
        prior.variance = adjusted_variance
        prior.mean = adjusted_mean
        return prior


class AdPredictor(BOPR, base.BinaryClassifier):
    """AdPredictor.
    """

    def __init__(self, initializer, surprise, epsilon):
        super().__init__(
            initializer=initializer,
            surprise=surprise,
            epsilon=epsilon
        )

    def fit_one(self, x, y):
        total_mean, total_variance = self._active_mean_variance(x)
        v, w = self._gaussian_correction(y*total_mean/total_variance)

        for i, xi in x.items():
            weight = self.weights[i]
            mean_delta = y * weight.variance / np.sqrt(total_variance) * v
            variance_multiplier = 1.0 - weight.variance / total_variance * w
            weight.mean += mean_delta
            weight.variance *= variance_multiplier
            self.weights[i] = weight

    def predict_proba_one(self, x):
        total_mean, total_variance = self._active_mean_variance(x)
        p = stats.norm.cdf(total_mean / total_variance)
        return {False: 1. - p, True: p}
