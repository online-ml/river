import math

from .. import stats

from . import base


__all__ = ['Gaussian']


class Gaussian(base.ContinuousDistribution):
    """Normal distribution with parameters mu and sigma.

    Example:

        ::

            >>> from creme import proba

            >>> p = proba.Gaussian().update(6).update(7)

            >>> p
            𝒩(μ=6.500, σ=0.707)

            >>> p.pdf(6.5)
            0.564189...

    """

    def __init__(self):
        self._var = stats.Var()

    @property
    def n_samples(self):
        return self._var.mean.n

    @property
    def n(self):
        return self.variance.mean.n

    @property
    def mu(self):
        return self._var.mean.get()

    @property
    def sigma(self):
        return self._var.get() ** 0.5

    @property
    def mode(self):
        return self.mu

    @property
    def mode(self):
        return self.mu

    def __str__(self):
        return f'𝒩(μ={self.mu:.3f}, σ={self.sigma:.3f})'

    def update(self, x):
        self._var.update(x)
        return self

    def pdf(self, x):
        var = self._var.get()
        if var:
            return math.exp((x - self.mu) ** 2 / (-2 * var)) / math.sqrt(math.tau * var)
        return 0.

    def cdf(self, x):
        try:
            return 0.5 * (1. + math.erf((x - self.mu) / (self.sigma * math.sqrt(2.))))
        except ZeroDivisionError:
            return 0.
