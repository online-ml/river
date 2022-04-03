import math
import random

from river import base, linear_model, stats


class ChebyshevUnderSampler(base.Wrapper, base.Regressor):
    """Under-sampling for imbalanced regression using Chebyshev's inequality.

    Chebyshev's inequality can be used to define the probability of target observations
    being frequent values (w.r.t. the distribution mean).

    Let $Y$ be a random variable with finite expected value $\\overline{y}$ and non-zero variance
    $\\sigma^2$. For any real number $t > 0$, the Chebyshev's inequality states that, for a wide class of
    unimodal probability distributions: $Pr(|y-\\overline{y}| \\ge t\\sigma) \\le \\dfrac{1}{t^2}$.

    Taking $t=\\dfrac{|y-\\overline{y}|}{\\sigma}$, and assuming $t > 1$, the Chebyshev’s inequality
    for an observation $y$ becomes: $P(|y - \\overline{y}|=t) = \\dfrac{\\sigma^2}{|y-\\overline{y}|}$.
    The reciprocal of this probability is used for under-sampling[^1] the most frequent cases. Extreme
    valued or rare cases have higher probabilities of selection, whereas the most frequent cases are
    likely to be discarded.


    Parameters
    ----------
    regressor
        The regression model that will receive the biased sample.
    seed
        Random seed to support reproducibility.

    Examples
    --------
    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import imblearn
    >>> from river import metrics
    >>> from river import preprocessing
    >>> from river import rules

    >>> model = (
    ...     preprocessing.StandardScaler() |
    ...     imblearn.ChebyshevUnderSampler(
    ...         regressor=rules.AMRules(
    ...             n_min=50, delta=0.01
    ...         ),
    ...         seed=42
    ...     )
    ... )

    >>> evaluate.progressive_val_score(
    ...     datasets.TrumpApproval(),
    ...     model,
    ...     metrics.MAE(),
    ...     print_every=500
    ... )
    [500] MAE: 1.84619
    [1,000] MAE: 1.516441
    MAE: 1.515879

    References
    ----------
    [^1]: Aminian, Ehsan, Rita P. Ribeiro, and João Gama. "Chebyshev approaches for imbalanced data
    streams regression models." Data Mining and Knowledge Discovery 35.6 (2021): 2389-2466.

    """

    def __init__(self, regressor: base.Regressor, seed: int = None):
        self.regressor = regressor
        self.seed = seed

        self._var = stats.Var()
        self._rng = random.Random(self.seed)

    @property
    def _wrapped_model(self):
        return self.regressor

    def predict_one(self, x):
        return self.regressor.predict_one(x)

    def learn_one(self, x, y, **kwargs):
        var = self._var.get()
        sd = var**0.5

        if sd > 0:
            mean = self._var.mean.get()
            dev = abs(y - mean)
            t = dev / sd
            if t > 1:
                prob_train = 1 - (var / (dev**2))
                p = self._rng.random()

                if p < prob_train:
                    self.regressor.learn_one(x, y, **kwargs)

        self._var.update(y)
        return self

    @classmethod
    def _unit_test_params(cls):
        yield {"regressor": linear_model.LinearRegression(), "seed": 42}


class ChebyshevOverSampler(base.Wrapper, base.Regressor):
    """Over-sampling for imbalanced regression using Chebyshev's inequality.

    Chebyshev's inequality can be used to define the probability of target observations
    being frequent values (w.r.t. the distribution mean).

    Let $Y$ be a random variable with finite expected value $\\overline{y}$ and non-zero variance
    $\\sigma^2$. For any real number $t > 0$, the Chebyshev's inequality states that, for a wide class of
    unimodal probability distributions: $Pr(|y-\\overline{y}| \\ge t\\sigma) \\le \\dfrac{1}{t^2}$.

    Taking $t=\\dfrac{|y-\\overline{y}|}{\\sigma}$, and assuming $t > 1$, the Chebyshev’s inequality
    for an observation $y$ becomes: $P(|y - \\overline{y}|=t) = \\dfrac{\\sigma^2}{|y-\\overline{y}|}$.

    Alternatively, one can use $t$ directly to estimate a frequency weight $\\kappa = \\lceil t\\rceil$
    and define an over-sampling strategy for extreme and rare target values[^1]. Each incoming instance is
    used $\\kappa$ times to update the underlying regressor, in case $t > 1$. Otherwise, the instance is
    ignored by the wrapped regression model.


    Parameters
    ----------
    regressor
        The regression model that will receive the biased sample.

    Examples
    --------
    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import imblearn
    >>> from river import metrics
    >>> from river import preprocessing
    >>> from river import rules

    >>> model = (
    ...     preprocessing.StandardScaler() |
    ...     imblearn.ChebyshevOverSampler(
    ...         regressor=rules.AMRules(
    ...             n_min=50, delta=0.01
    ...         )
    ...     )
    ... )

    >>> evaluate.progressive_val_score(
    ...     datasets.TrumpApproval(),
    ...     model,
    ...     metrics.MAE(),
    ...     print_every=500
    ... )
    [500] MAE: 2.131883
    [1,000] MAE: 1.496747
    MAE: 1.496013

    References
    ----------
    [^1]: Aminian, Ehsan, Rita P. Ribeiro, and João Gama. "Chebyshev approaches for imbalanced data
    streams regression models." Data Mining and Knowledge Discovery 35.6 (2021): 2389-2466.

    """

    def __init__(self, regressor: base.Regressor):
        self.regressor = regressor

        self._var = stats.Var()

    @property
    def _wrapped_model(self):
        return self.regressor

    def predict_one(self, x):
        return self.regressor.predict_one(x)

    def learn_one(self, x, y, **kwargs):
        var = self._var.get()
        sd = var**0.5

        if sd > 0:
            mean = self._var.mean.get()
            dev = abs(y - mean)
            t = dev / sd

            if t > 1:
                kappa = int(math.ceil(t))

                for k in range(kappa):
                    self.regressor.learn_one(x, y, **kwargs)

        self._var.update(y)
        return self

    @classmethod
    def _unit_test_params(cls):
        yield {"regressor": linear_model.LinearRegression()}
