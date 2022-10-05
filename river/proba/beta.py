import collections
import math
import typing

from river.proba import base

__all__ = ["Beta"]


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

    References
    ----------
    [^1]: [What is the intuition behind beta distribution?](https://stats.stackexchange.com/questions/47771/what-is-the-intuition-behind-beta-distribution)

    """

    def __init__(self, alpha: int = 1, beta: int = 1):
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
        return math.gamma(self.alpha + self.beta)
        return (
            math.gamma(self.alpha + self.beta) * p ** (self.alpha - 1) * (1 - p) ** (self.beta - 1) /
            (math.gamma(self.alpha) * math.gamma(self.beta))
        )

    def sample(self):
        from scipy import stats
        return stats.beta.rvs(self.alpha, self.beta)
