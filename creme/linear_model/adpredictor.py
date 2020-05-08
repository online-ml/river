import collections
import numbers
import numpy as np 
from scipy import stats
from .. import base 
from .. import optim
from .. import utils



__all__ = [
    'AdPredictor'
]


class NormalPrior(optim.initializers.Initializer):
    """Initializer used to set beliefs over weights of a model.

   Initializer which represents a gaussian belief over a weight i.e. a normal prior with parameters μ ``mean`` and σ2 ``variance``. 
   The ``mean`` is calculated taking the predictive posterior closed form inverse of the prior probability value.

    Parameters:
        prior_probability (float): Prior probability on the feature weights prior_probability = P(y=1 | x, weights).
        beta (float): Scale of the inverse link function (standardized cumulative normal distribution function).
        n_features (int): Number of features.

    Example:

        ::

            >>> from creme import linear_model
            >>> from creme import datasets

            >>> X_y = datasets.Phishing()

            >>> init = linear_model.adpredictor.NormalPrior(beta=0.05, n_features=X_y.n_features, prior_probability=0.6)

            >>> print(init)
            Biased normal prior on weights 𝒩(μ=2.281, σ2=1.000)


    Attributes:
        mean (float): Mean of the gaussian belief over the weight.
        variance (float): Variance of the gaussian belief over the weight.

    """
    def __init__(self, beta, n_features, prior_probability=None):
        self.beta = beta
        self.n_features = n_features
        self.prior_probability = prior_probability
        self.mean = 0 if prior_probability is None else stats.norm.ppf(prior_probability) * (beta ** 2 + n_features)
        self.variance = 1.0

    def __str__(self):
        return f'Unbiased normal prior on weights 𝒩(μ={self.mean:.3f}, σ2={self.variance:.3f})' if self.prior_probability is None else f'Biased normal prior on weights 𝒩(μ={self.mean:.3f}, σ2={self.variance:.3f})'

    def __call__(self):
        return self


class BOPR:
    """Bayesian Online Probit Regression Model.

    Parameters:
        initializer (linear_model.adpredictor.NormalPrior): The initializer used to set beliefs over the weights.
        surprise (float): Clipping value of the learning step function parameter ensure numerical stability.
        epsilon (float): Noise parameter of the dynamic correction process.

    Attributes:
        weights (collections.defaultdict): The current gaussian belief over the weights.
        
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
        """TODO 
        """
        t = utils.math.clamp(t, minimum=-self.surprise, maximum=self.surprise)
        v = stats.norm.pdf(t)/stats.norm.cdf(t)
        w = v * (v + t)
        return (v, w)

    def _apply_dynamics(self, weight):
        """TODO
        """
        prior = self.initializer
        adjusted_variance = weight.variance * prior.variance / ((1.0 - self.epsilon) * prior.variance + self.epsilon * weight.variance)
        adjusted_mean = adjusted_variance * ((1.0 - self.epsilon) * weight.mean / weight.variance + self.epsilon * prior.mean / prior.variance)
        prior.variance = adjusted_variance
        prior.mean = adjusted_mean
        return prior

    @staticmethod
    def _target_encoding(y):
        assert isinstance(y, numbers.Number)
        return 1.0 if y == 1 else -1.0


class AdPredictor(BOPR, base.BinaryClassifier):
    """AdPredictor is a bayesian online probit regression model devised for search advertising click through rate (CTR) prediction.

    AdPredictor

    Parameters:
        initializer (linear_model.adpredictor.NormalPrior): The initializer used to set beliefs over the weights.
        surprise (float): Clipping value of the learning step function parameter ensure numerical stability.
        epsilon (float): Noise parameter of the dynamic correction process.

    Attributes:
        weights (collections.defaultdict): The current gaussian belief over the weights.

    Example:

        ::

            >>> X_y = datasets.Phishing()

            >>> model = (
            ...     compose.Select('age_of_domain','ip_in_url')|
            ...     preprocessing.OneHotEncoder('age_of_domain')|
            ...     linear_model.AdPredictor(initializer=NormalPrior(beta=0.1, n_features=X_y.n_features, prior_probability=0.6), surprise=5, epsilon=0.001)
            ... )

            >>> metric = metrics.ROCAUC()

            >>> model_selection.progressive_val_score(X_y, model, metric)
            ROCAUC: 0.540405

    Notes:
        In (1) the application is focused on discrete multi-valued features where the model is fitted on 1-in-N encoding of the features. 
        Using preprocessing is advised.

    References:
        1. `Graepel, Thore, et al. "Web-scale bayesian click-through rate prediction for sponsored search advertising in microsoft's bing search engine." Omnipress, 2010. <https://discovery.ucl.ac.uk/id/eprint/1395202/1/Graepel_901%255B1%255D.pdf>`
        2. `He, Xinran, et al. "Practical lessons from predicting clicks on ads at facebook." Proceedings of the Eighth International Workshop on Data Mining for Online Advertising. 2014. <https://dl.acm.org/doi/pdf/10.1145/2648584.2648589>'
        3. `Tulloch, Andrew. "Online Learning with Microsoft's AdPredictor algorithm" <http://tullo.ch/articles/online-learning-with-adpredictor/>`

    """

    def __init__(self, initializer, surprise, epsilon):
        super().__init__(
            initializer=initializer,
            surprise=surprise,
            epsilon=epsilon
        )

    def fit_one(self, x, y):
        total_mean, total_variance = self._active_mean_variance(x)
        y = BOPR._target_encoding(y)
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
