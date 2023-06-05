from __future__ import annotations

import math

import scipy.special

from river.proba import base

__all__ = ["Beta"]


def _beta_func(a, b):
    """

    A naive implementation with (math.gamma(a) + math.gamma(b)) / math.gamma(a + b) would
    overflow for large values of a and b.

    See https://malishoaib.wordpress.com/2014/04/15/the-beautiful-beta-functions-in-raw-python/
    for more details.

    """
    return math.exp(math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))


class Beta(base.ContinuousDistribution):
    """Beta distribution for binary data.

    A Beta distribution is very similar to a Bernoulli distribution in that it counts occurrences
    of boolean events. The differences lies in what is being measured. A Binomial distribution
    models the probability of an event occurring, whereas a Beta distribution models the
    probability distribution itself. In other words, it's a probability distribution over
    probability distributions.

    Parameters
    ----------
    alpha
        Initial alpha parameter.
    beta
        Initial beta parameter.
    seed
        Random number generator seed for reproducibility.

    Examples
    --------

    >>> from river import proba

    >>> successes = 81
    >>> failures = 219
    >>> beta = proba.Beta(successes, failures)

    >>> beta(.21), beta(.35)
    (0.867..., 0.165...)

    >>> for success in range(100):
    ...     beta = beta.update(True)
    >>> for failure in range(200):
    ...     beta = beta.update(False)

    >>> beta(.21), beta(.35)
    (2.525...e-05, 0.841...)

    >>> beta.cdf(.35)
    0.994168...

    References
    ----------
    [^1]: [What is the intuition behind beta distribution?](https://stats.stackexchange.com/questions/47771/what-is-the-intuition-behind-beta-distribution)

    """

    def __init__(self, alpha: int = 1, beta: int = 1, seed: int | None = None):
        super().__init__(seed)
        self.alpha = alpha
        self.beta = beta
        self._alpha = alpha
        self._beta = beta

    @property
    def n_samples(self):
        return self._alpha - self.alpha + self._beta - self.beta

    def update(self, x):
        if x:
            self.alpha += 1
        else:
            self.beta += 1
        return self

    def revert(self, x):
        if x:
            self.alpha -= 1
        else:
            self.beta -= 1
        return self

    def __call__(self, p: float):
        return (
            p ** (self.alpha - 1) * (1 - p) ** (self.beta - 1) / _beta_func(self.alpha, self.beta)
        )

    def sample(self):
        return self._rng.betavariate(self.alpha, self.beta)

    @property
    def mode(self):
        try:
            return (self.alpha - 1) / (self.alpha + self.beta - 2)
        except ZeroDivisionError:
            return 0.5

    def cdf(self, x):
        return scipy.special.betainc(self.alpha, self.beta, x)
