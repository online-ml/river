from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import multivariate_normal

from river import covariance, stats
from river.proba import base

__all__ = ["Gaussian", "MultivariateGaussian"]


class Gaussian(base.ContinuousDistribution):
    """Normal distribution with parameters mu and sigma.

    Parameters
    ----------
    seed
        Random number generator seed for reproducibility.

    Examples
    --------

    >>> from river import proba

    >>> p = proba.Gaussian().update(6).update(7)

    >>> p
    𝒩(μ=6.500, σ=0.707)

    >>> p(6.5)
    0.564189

    >>> p.revert(7)
    𝒩(μ=6.000, σ=0.000)

    """

    def __init__(self, seed=None):
        super().__init__(seed)
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

    def __repr__(self):
        return f"𝒩(μ={self.mu:.3f}, σ={self.sigma:.3f})"

    def update(self, x, w=1.0):
        self._var.update(x, w)
        return self

    def revert(self, x, w=1.0):
        self._var.revert(x, w)
        return self

    def __call__(self, x):
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

    def sample(self):
        return self._rng.gauss(self.mu, self.sigma)

    @property
    def mode(self):
        return self.mu


class MultivariateGaussian(base.MultivariateContinuousDistribution):
    """Multivariate normal distribution with parameters mu and var.

    Parameters
    ----------
    seed
        Random number generator seed for reproducibility.

    Examples
    --------

    >>> import numpy as np
    >>> import pandas as pd
    >>> from river import proba

    >>> np.random.seed(42)
    >>> X = pd.DataFrame(
    ...     np.random.random((8, 3)),
    ...     columns=["red", "green", "blue"]
    ... )
    >>> X
            red     green      blue
    0  0.374540  0.950714  0.731994
    1  0.598658  0.156019  0.155995
    2  0.058084  0.866176  0.601115
    3  0.708073  0.020584  0.969910
    4  0.832443  0.212339  0.181825
    5  0.183405  0.304242  0.524756
    6  0.431945  0.291229  0.611853
    7  0.139494  0.292145  0.366362


    >>> p = proba.MultivariateGaussian(seed=42)
    >>> p.n_samples
    0.0

    >>> for x in X.to_dict(orient="records"):
    ...     p = p.update(x)
    >>> p.var
               blue     green       red
    blue   0.076119  0.020292 -0.010128
    green  0.020292  0.112931 -0.053268
    red   -0.010128 -0.053268  0.078961

    Retrieving current state in nice format is simple
    >>> p
    𝒩(
        μ=(0.518, 0.387, 0.416),
        σ^2=(
            [ 0.076  0.020 -0.010]
            [ 0.020  0.113 -0.053]
            [-0.010 -0.053  0.079]
        )
    )

    To retrieve number of samples and mode:

    >>> p.n_samples
    8.0
    >>> p.mode
    {'blue': 0.5179..., 'green': 0.3866..., 'red': 0.4158...}

    To retrieve the PDF and CDF:

    >>> p(x)
    0.97967...
    >>> p.cdf(x)
    0.00787...

    To sample data from distribution:

    >>> p.sample()
    {'blue': -0.179..., 'green': -0.051..., 'red': 0.376...}

    MultivariateGaussian works with `utils.Rolling`:

    >>> from river import utils

    >>> p = utils.Rolling(MultivariateGaussian(), window_size=5)
    >>> for x in X.to_dict(orient="records"):
    ...     p = p.update(x)
    >>> p.var
               blue     green       red
    blue   0.087062 -0.022873  0.007765
    green -0.022873  0.014279 -0.025181
    red    0.007765 -0.025181  0.095066

    MultivariateGaussian works with `utils.TimeRolling`:

    >>> from datetime import datetime as dt, timedelta as td
    >>> X.index = [dt(2023, 3, 28, 0, 0, 0) + td(seconds=x) for x in range(8)]
    >>> p = utils.TimeRolling(MultivariateGaussian(), period=td(seconds=5))
    >>> for t, x in X.iterrows():
    ...     p = p.update(x.to_dict(), t=t)
    >>> p.var
               blue     green       red
    blue   0.087062 -0.022873  0.007765
    green -0.022873  0.014279 -0.025181
    red    0.007765 -0.025181  0.095066

    Variance on diagonal is consistent with `proba.Gaussian`.

    >>> multi = proba.MultivariateGaussian()
    >>> single = proba.Gaussian()
    >>> for x in X.to_dict(orient='records'):
    ...     multi = multi.update(x)
    ...     single = single.update(x['blue'])
    >>> multi.mu['blue'] == single.mu
    True
    >>> multi.sigma['blue']['blue'] == single.sigma
    True

    """

    def __init__(self, seed=None):
        super().__init__(seed)
        self._var = covariance.EmpiricalCovariance(ddof=1)

    # TODO: add method _from_state to initialize model (for warm starting)

    @property
    def n_samples(self) -> float:
        if not self._var.matrix:
            return 0.0
        else:
            return list(self._var.matrix.values())[-1].mean.n

    @property
    def mu(self) -> dict:
        """The mean value of the distribution."""
        return {
            key1: values.mean.get()
            for (key1, key2), values in sorted(self._var.matrix.items())
            if key1 == key2
        }

    @property
    def var(self) -> pd.DataFrame:
        """The variance of the distribution."""
        variables = sorted(list({var for cov in self._var.matrix.keys() for var in cov}))
        # Initialize the covariance matrix array
        cov_array = np.zeros((len(variables), len(variables)))

        # Fill in the covariance matrix array
        for i in range(len(variables)):
            for j in range(i, len(variables)):
                if i == j:
                    # Fill in the diagonal with variances
                    cov_array[i, j] = self._var[(variables[i], variables[j])].get()
                else:
                    # Fill in the off-diagonal with covariances
                    cov_array[i, j] = self._var[(variables[i], variables[j])].get()
                    cov_array[j, i] = self._var[(variables[i], variables[j])].get()

        cov_array = pd.DataFrame(cov_array, index=variables, columns=variables)
        return cov_array

    @property
    def sigma(self) -> pd.DataFrame:
        """The standard deviation of the distribution."""
        return self.var**0.5

    def __repr__(self):
        mu_str = ", ".join(f"{m:.3f}" for m in self.mu.values())
        var_str = self.var.to_string(float_format="{:0.3f}".format, header=False, index=False)
        var_str = "        [" + var_str.replace("\n", "]\n        [") + "]"
        return f"𝒩(\n    μ=({mu_str}),\n    σ^2=(\n{var_str}\n    )\n)"

    def update(self, x):
        # TODO: add support for weigthed samples
        self._var.update(x)
        return self

    def revert(self, x):
        # TODO: add support for weigthed samples
        self._var.revert(x)
        return self

    def __call__(self, x: dict[str, float]):
        """PDF(x) method."""
        x_ = [x[i] for i in self.mu]
        var = self.var
        if var is not None:
            try:
                pdf_ = multivariate_normal([*self.mu.values()], var).pdf(x_)
                return float(pdf_)
            # TODO: validate occurence of ValueError
            # The input matrix must be symmetric positive semidefinite.
            except ValueError:  # pragma: no cover
                return 0.0
            # TODO: validate occurence of OverflowError
            except OverflowError:  # pragma: no cover
                return 0.0
        return 0.0  # pragma: no cover

    def cdf(self, x: dict[str, float]):
        x_ = list(x.values())
        cdf_ = multivariate_normal(
            [self.mu[i] for i in x],
            self.var,
            allow_singular=True,
        ).cdf(x_)
        return float(cdf_)

    def sample(self) -> dict[str, float]:
        sample_ = map(
            float,
            multivariate_normal(
                [*self.mu.values()], self.var, seed=self._rng.randint(0, 2**10)
            ).rvs(),
        )
        return dict(zip(self.mu.keys(), sample_))

    @property
    def mode(self) -> dict:
        return self.mu


@pytest.mark.parametrize(
    "p",
    [
        pytest.param(
            p,
            id=f"{p=}",
        )
        for p in [1, 3, 5]
    ],
)
def test_univariate_multivariate_consistency(p):
    X = pd.DataFrame(np.random.random((30, p)), columns=range(p))

    multi = MultivariateGaussian()
    single = {c: Gaussian() for c in X.columns}

    for x in X.to_dict(orient="records"):
        multi = multi.update(x)
        for c, s in single.items():
            s.update(x[c])

    for c in X.columns:
        assert math.isclose(multi.mu[c], single[c].mu)
        assert math.isclose(multi.sigma[c][c], single[c].sigma)
