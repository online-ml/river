from __future__ import annotations

import copy

import numpy as np

from river import stats


class Mean(stats.base.Univariate):
    """Running mean.

    Attributes
    ----------
    n : float
        The current sum of weights. If each passed weight was 1, then this is equal to the number
        of seen observations.

    Examples
    --------

    >>> from river import stats

    >>> X = [-5, -3, -1, 1, 3, 5]
    >>> mean = stats.Mean()
    >>> for x in X:
    ...     print(mean.update(x).get())
    -5.0
    -4.0
    -3.0
    -2.0
    -1.0
    0.0

    You can calculate a rolling average by wrapping a `utils.Rolling` around:

    >>> from river import utils

    >>> X = [1, 2, 3, 4, 5, 6]
    >>> rmean = utils.Rolling(stats.Mean(), window_size=2)

    >>> for x in X:
    ...     print(rmean.update(x).get())
    1.0
    1.5
    2.5
    3.5
    4.5
    5.5

    References
    ----------
    [^1]: [West, D. H. D. (1979). Updating mean and variance estimates: An improved method. Communications of the ACM, 22(9), 532-535.](https://dl.acm.org/doi/10.1145/359146.359153)
    [^2]: [Finch, T., 2009. Incremental calculation of weighted mean and variance. University of Cambridge, 4(11-5), pp.41-42.](https://fanf2.user.srcf.net/hermes/doc/antiforgery/stats.pdf)
    [^3]: [Chan, T.F., Golub, G.H. and LeVeque, R.J., 1983. Algorithms for computing the sample variance: Analysis and recommendations. The American Statistician, 37(3), pp.242-247.](https://amstat.tandfonline.com/doi/abs/10.1080/00031305.1983.10483115)

    """

    def __init__(self):
        self.n = 0
        self._mean = 0.0

    def update(self, x, w=1.0):
        self.n += w
        self._mean += (w / self.n) * (x - self._mean)
        return self

    def update_many(self, X: np.ndarray):
        a = self.n / (self.n + len(X))
        b = len(X) / (self.n + len(X))
        self._mean = a * self._mean + b * np.mean(X)
        self.n += len(X)
        return self

    def revert(self, x, w=1.0):
        self.n -= w
        if self.n < 0:
            raise ValueError("Cannot go below 0")
        elif self.n == 0:
            self._mean = 0.0
        else:
            self._mean -= (w / self.n) * (x - self._mean)
        return self

    def get(self):
        return self._mean

    @classmethod
    def _from_state(cls, n, mean):
        new = cls()
        new.n = n
        new._mean = mean

        return new

    def __iadd__(self, other):
        old_n = self.n
        self.n += other.n
        self._mean = (old_n * self._mean + other.n * other.get()) / self.n
        return self

    def __add__(self, other):
        result = copy.deepcopy(self)
        result += other
        return result

    def __isub__(self, other):
        old_n = self.n
        self.n -= other.n

        if self.n > 0:
            self._mean = (old_n * self._mean - other.n * other._mean) / self.n
        else:
            self.n = 0.0
            self._mean = 0.0
        return self

    def __sub__(self, other):
        result = copy.deepcopy(self)
        result -= other
        return result


class BayesianMean(stats.base.Univariate):
    """Estimates a mean using outside information.

    Parameters
    ----------
    prior
    prior_weight

    References
    ----------
    [^1]: [Additive smoothing](https://www.wikiwand.com/en/Additive_smoothing)
    [^2]: [Bayesian average](https://www.wikiwand.com/en/Bayesian_average)
    [^3]: [Practical example of Bayes estimators](https://www.wikiwand.com/en/Bayes_estimator#/Practical_example_of_Bayes_estimators)

    """

    def __init__(self, prior: float, prior_weight: float):
        self.prior = prior
        self.prior_weight = prior_weight
        self._mean = Mean()

    @property
    def name(self):
        return "bayes_mean"

    def update(self, x):
        self._mean.update(x)
        return self

    def revert(self, x):
        self._mean.revert(x)
        return self

    def get(self):
        # Uses the notation from https://www.wikiwand.com/en/Bayes_estimator#/Practical_example_of_Bayes_estimators
        R = self._mean.get()
        v = self._mean.n
        m = self.prior_weight
        C = self.prior

        return (R * v + C * m) / (v + m)
