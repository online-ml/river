from __future__ import annotations

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
    likely to be discarded. Still, frequent cases have a small chance of being selected (controlled via
    the `sp` parameter) in case few rare instances were observed.

    Parameters
    ----------
    regressor
        The regression model that will receive the biased sample.
    sp
        Second chance probability. Even if an example is not initially selected for training, it still has
        a small chance of being selected in case the number of rare case observed so far is small.
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
    ...             n_min=50, delta=0.01,
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
    [500] MAE: 1.787162
    [1,000] MAE: 1.515711
    [1,001] MAE: 1.515236
    MAE: 1.515236

    References
    ----------
    [^1]: Aminian, Ehsan, Rita P. Ribeiro, and João Gama. "Chebyshev approaches for imbalanced data
    streams regression models." Data Mining and Knowledge Discovery 35.6 (2021): 2389-2466.

    """

    def __init__(self, regressor: base.Regressor, sp: float = 0.15, seed: int | None = None):
        self.regressor = regressor
        self.sp = sp
        self.seed = seed

        self._var = stats.Var()
        self._rng = random.Random(self.seed)

        self._freq_c = 0
        self._rare_c = 0

    @property
    def _wrapped_model(self):
        return self.regressor

    def predict_one(self, x, **kwargs):
        return self.regressor.predict_one(x, **kwargs)

    def learn_one(self, x, y, **kwargs):
        self._var.update(y)
        sd = self._var.get() ** 0.5

        if sd > 0:
            mean = self._var.mean.get()
            dev = abs(y - mean)  # noqa
            t = dev / sd

            # Small values for rare cases and 1 for frequent cases
            prob_threshold = 1 / (t * t) if t > 1 else 1
            p = self._rng.random()

            if p >= prob_threshold:
                self.regressor.learn_one(x, y, **kwargs)
                self._rare_c += 1
            elif self._freq_c < self._rare_c and p <= self.sp:
                self.regressor.learn_one(x, y, **kwargs)
                self._freq_c += 1
        else:
            self.regressor.learn_one(x, y, **kwargs)

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
    used $\\kappa$ times to update the underlying regressor. Frequent target values contribute only once
    to the underlying regressor, whereas rares cases are used multiple times for training.

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
    [500] MAE: 1.673902
    [1,000] MAE: 1.743046
    [1,001] MAE: 1.741335
    MAE: 1.741335

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

    def predict_one(self, x, **kwargs):
        return self.regressor.predict_one(x, **kwargs)

    def learn_one(self, x, y, **kwargs):
        self._var.update(y)
        var = self._var.get()
        sd = var**0.5

        if sd > 0:
            mean = self._var.mean.get()
            dev = abs(y - mean)  # noqa
            t = dev / sd

            kappa = int(math.ceil(t))

            for k in range(kappa):
                self.regressor.learn_one(x, y, **kwargs)
        else:
            self.regressor.learn_one(x, y, **kwargs)

        return self

    @classmethod
    def _unit_test_params(cls):
        yield {"regressor": linear_model.LinearRegression()}
