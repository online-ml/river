from __future__ import annotations

import math

from scipy.special import ndtri

from river import base, utils

__all__ = ["AdPredictor"]

# Surprise clipping, exactly as in the original AdPredictor/TrueSkill code, to
# avoid the ~0/~0 blow-up of v(t) = pdf(t) / cdf(t) deep in the tails.
MAX_SURPRISE = 5.0

_BIAS_KEY = "__bias__"


def _active(x: dict) -> list:
    """The keys of `x` with a truthy value, i.e. the active binary indicators."""
    return [f for f, v in x.items() if v]


def _v_w(t: float) -> tuple[float, float]:
    """Truncated-Gaussian corrections v(t)=N(t)/Phi(t), w(t)=v(t)*(v(t)+t)."""
    t = max(-MAX_SURPRISE, min(MAX_SURPRISE, t))
    v = utils.math.norm_pdf(t) / utils.math.norm_cdf(t)
    w = v * (v + t)
    return v, w


class AdPredictor(base.Classifier):
    """Bayesian online probit regression for click-through-rate prediction.

    AdPredictor, used at some point by Microsoft for CTR prediction in Bing's sponsored search
    [^1], keeps a Gaussian belief over each feature weight rather than a point estimate. It
    predicts via a probit link and learns in a single pass, scaling each weight's step size by its
    own uncertainty. It shines on the sparse, high-cardinality, categorical data of ad logs: it
    yields well-calibrated probabilities and exposes uncertainty for exploration. Like the other
    linear models, its cost per example scales only with the number of active features. Plain
    logistic regression is simpler and just as good on dense, low-dimensional numeric data.

    Features are expected to be a sparse active set: a key in `x` is active when its value is
    truthy (its magnitude is otherwise ignored), so one-hot encode or bucket
    inputs. Use `preprocessing.OneHotEncoder(drop_zeros=True)` to feed only the
    present categories.

    Parameters
    ----------
    beta
        Standard deviation of the per-example label noise; the prediction step
        size shrinks as `beta` grows.
    prior_probability
        Base-rate CTR used to initialise the bias weight, so the model predicts
        this value before seeing any data. The bias is calibrated lazily from the
        number of active features in the first observed example.
    epsilon
        Variance-dynamics rate in [0, 1). Each update nudges weights a fraction
        `epsilon` of the way back toward the unit prior, preventing variances
        from collapsing and letting the model track drift. Set to 0 to disable.

    Examples
    --------

    >>> from river import linear_model

    >>> model = linear_model.AdPredictor(beta=0.1, prior_probability=0.5)

    With no data seen yet, the model predicts the base rate:

    >>> model.predict_proba_one({"a": 1, "b": 1})[True]
    0.5

    After repeatedly seeing the same clicked impression, its belief moves up:

    >>> for _ in range(50):
    ...     model.learn_one({"a": 1, "b": 1}, True)
    >>> model.predict_proba_one({"a": 1, "b": 1})[True] > 0.9
    True

    On a real ad-click stream, one-hot encode the fields into a sparse active set
    and evaluate with progressive validation:

    >>> from river import datasets, metrics, preprocessing

    >>> dataset = datasets.CriteoAds()
    >>> model = preprocessing.OneHotEncoder(drop_zeros=True) | linear_model.AdPredictor()
    >>> metric = metrics.ROCAUC()

    >>> for x, y in dataset.take(10_000):
    ...     metric.update(y, model.predict_proba_one(x))
    ...     model.learn_one(x, y)
    >>> metric
    ROCAUC: 64.95%

    References
    ----------
    [^1]: Graepel, T., Candela, J.Q., Borchert, T. and Herbrich, R., 2010.
        [Web-scale Bayesian click-through rate prediction for sponsored search advertising in Microsoft's Bing search engine](https://quinonero.net/Publications/AdPredictorICML2010-final.pdf).
        ICML 2010.

    """

    def __init__(
        self,
        beta: float = 0.1,
        prior_probability: float = 0.5,
        epsilon: float = 0.05,
    ):
        self.beta = beta
        self.prior_probability = prior_probability
        self.epsilon = epsilon

        self.means: dict = {}
        self.variances: dict = {}

    def _init_bias(self, n_active: int) -> None:
        # The bias is an always-active weight, initialised lazily on the first
        # observation. Its mean is set so that, with the `n_active` zero-mean
        # unit-variance weights of that example also active, the probit prediction
        # equals `prior_probability` -- this estimates on the fly what used to be
        # the `num_features` parameter.
        if _BIAS_KEY not in self.means:
            self.means[_BIAS_KEY] = float(ndtri(self.prior_probability)) * math.sqrt(
                self.beta**2 + n_active
            )
            self.variances[_BIAS_KEY] = 1.0

    def _total_mean_variance(self, features) -> tuple[float, float]:
        total_mean = self.means[_BIAS_KEY]
        total_variance = self.variances[_BIAS_KEY] + self.beta**2
        for f in features:
            total_mean += self.means.get(f, 0.0)
            total_variance += self.variances.get(f, 1.0)
        return total_mean, total_variance

    def predict_proba_one(self, x):
        active = _active(x)
        self._init_bias(len(active))
        total_mean, total_variance = self._total_mean_variance(active)
        p = utils.math.norm_cdf(total_mean / math.sqrt(total_variance))
        p = min(1.0 - 1e-12, max(1e-12, p))
        return {False: 1.0 - p, True: p}

    def learn_one(self, x, y):
        features = _active(x)
        self._init_bias(len(features))
        sign = 1.0 if y else -1.0
        total_mean, total_variance = self._total_mean_variance(features)
        std = math.sqrt(total_variance)
        v, w = _v_w(sign * total_mean / std)

        for f in (_BIAS_KEY, *features):
            mean = self.means.get(f, 0.0)
            variance = self.variances.get(f, 1.0)

            mean = mean + sign * variance / std * v
            variance = variance * (1.0 - variance / total_variance * w)

            self.means[f], self.variances[f] = self._apply_dynamics(mean, variance)

    def _apply_dynamics(self, mean: float, variance: float) -> tuple[float, float]:
        """Nudge a weight back toward the unit prior N(0, 1) by a factor epsilon."""
        eps = self.epsilon
        if eps == 0.0:
            return mean, variance
        prior_variance = 1.0
        adjusted_variance = (
            variance * prior_variance / ((1.0 - eps) * prior_variance + eps * variance)
        )
        # The prior mean is 0, so its contribution to the adjusted mean drops out.
        adjusted_mean = adjusted_variance * ((1.0 - eps) * mean / variance)
        return adjusted_mean, adjusted_variance

    @classmethod
    def _unit_test_params(cls):
        yield {"beta": 0.1, "prior_probability": 0.5, "epsilon": 0.05}
