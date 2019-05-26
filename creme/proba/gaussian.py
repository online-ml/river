import math

from .. import stats

from . import base


class Gaussian(base.Distribution):
    """Normal distribution with parameters mu and sigma.

    Example:

        ::

            >>> from creme import proba

            >>> p = proba.Gaussian().update(6)
            >>> p.mode()
            6.0

            >>> p.update(7).mode()
            6.5

            >>> p
            𝒩(μ=6.500, σ=0.707)

            >>> p.proba_of(6.5)
            0.564189...

    """

    def __init__(self):
        self.variance = stats.Var()

    @property
    def mu(self):
        return self.variance.mean.get()

    @property
    def sigma(self):
        return self.variance.get() ** 0.5

    def __str__(self):
        return f'𝒩(μ={self.mu:.3f}, σ={self.sigma:.3f})'

    def __repr__(self):
        return str(self)

    def update(self, x):
        self.variance.update(x)
        return self

    def mode(self):
        return self.mu

    def proba_of(self, x):
        variance = self.variance.get()

        if variance == 0:
            return 0

        return math.exp((x - self.mu) ** 2 / (-2 * variance)) / math.sqrt(math.tau * variance)
