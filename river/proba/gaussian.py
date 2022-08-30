import math

from river import stats

from . import base

__all__ = ["Gaussian"]


class Gaussian(base.ContinuousDistribution):
    """Normal distribution with parameters mu and sigma.

    Examples
    --------

    >>> from river import proba

    >>> p = proba.Gaussian().update(6).update(7)

    >>> p
    𝒩(μ=6.500, σ=0.707)

    >>> p.pdf(6.5)
    0.564189

    >>> p.revert(7)
    𝒩(μ=6.000, σ=0.000)

    """

    def __init__(self):
        self._var = stats.Var(ddof=1)

    @classmethod
    def _from_state(cls, n, m, sig, ddof):
        new = cls()
        new._var = stats.Var._from_state(n, m, sig, ddof=ddof)
        return new

    @property
    def n_samples(self):
        return self._var.mean.n

    @property
    def mu(self):
        return self._var.mean.get()

    @property
    def sigma(self):
        return self._var.get() ** 0.5

    @property
    def mode(self):
        return self.mu

    def __str__(self):
        return f"𝒩(μ={self.mu:.3f}, σ={self.sigma:.3f})"

    def update(self, x, w=1.0):
        self._var.update(x, w)
        return self

    def revert(self, x, w=1.0):
        self._var.revert(x, w)
        return self

    def pdf(self, x):
        var = self._var.get()
        if var:
            try:
                return math.exp((x - self.mu) ** 2 / (-2 * var)) / math.sqrt(math.tau * var)
            except ValueError:
                return 0.0
            except OverflowError:
                return 0.0
        return 0.0

    def cdf(self, x):
        try:
            return 0.5 * (1.0 + math.erf((x - self.mu) / (self.sigma * math.sqrt(2.0))))
        except ZeroDivisionError:
            return 0.0
