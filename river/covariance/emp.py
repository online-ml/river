import abc
import itertools

import numpy as np
import pandas as pd
import scipy as sp

from river import stats, utils


class SymmetricMatrix(abc.ABC):

    @property
    @abc.abstractmethod
    def matrix(self):
        ...

    def __getitem__(self, key):
        """

        A covariance matrix is symmetric. For ease of use we make the __getitem__ method symmetric.

        """
        x, y = key
        try:
            return self.matrix[x, y]
        except KeyError:
            return self.matrix[y, x]

    def __repr__(self):

        names = sorted(set(i for i, _ in self.matrix))

        headers = [""] + list(map(str, names))
        columns = [headers[1:]]
        for col in names:
            column = []
            for row in names:
                try:
                    column.append(f"{self[row, col]:.3f}")
                except KeyError:
                    column.append("")
            columns.append(column)

        return utils.pretty.print_table(headers, columns)


class EmpiricalCovariance(SymmetricMatrix):
    """Empirical covariance matrix.

    Parameters
    ----------
    ddof
        Delta Degrees of Freedom.

    Examples
    --------

    >>> import numpy as np
    >>> import pandas as pd
    >>> from river import covariance

    >>> np.random.seed(42)
    >>> X = pd.DataFrame(np.random.random((8, 3)), columns=["red", "green", "blue"])
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

    >>> cov = covariance.EmpiricalCovariance()
    >>> for x in X.to_dict(orient="records"):
    ...     cov = cov.update(x)
    >>> cov
            blue     green    red
     blue    0.076    0.020   -0.010
    green    0.020    0.113   -0.053
      red   -0.010   -0.053    0.079

    There is also an `update_many` method to process mini-batches. The results are identical.

    >>> cov = covariance.EmpiricalCovariance()
    >>> cov = cov.update_many(X)
    >>> cov
            blue     green    red
     blue    0.076    0.020   -0.010
    green    0.020    0.113   -0.053
      red   -0.010   -0.053    0.079

    The covariances are stored in a dictionary, meaning any one of them can be accessed as such:

    >>> cov["blue", "green"]
    0.020291...

    Diagonal entries are variances:

    >>> cov["blue", "blue"]
    0.076119...

    """

    def __init__(self, ddof=1):
        self.ddof = ddof
        self._occ = {}
        self._loc = {}
        self._cov = {}

    @property
    def matrix(self):
        return self._cov

    def update(self, x: dict):
        """Update with a single sample.

        Parameters
        ----------
        x
            A sample.

        """

        # dict -> numpy
        x_vec = np.array(list(x.values()))
        occ, loc, cov = self._get_stats(variables=x.keys())

        # update formulas
        diff = x_vec - loc
        occ += 1
        loc += diff / occ
        cov += (np.outer(diff, x_vec - loc) - cov) / np.maximum(occ - self.ddof, 1)

        # numpy -> dict
        self._set_stats(x.keys(), occ, loc, cov)

        return self

    def revert(self, x: dict):
        """Downdate with a single sample.

        Parameters
        ----------
        x
            A sample.

        """

        # dict -> numpy
        x_vec = np.array(list(x.values()))
        occ, loc, cov = self._get_stats(variables=x.keys())

        # update formulas
        diff = x_vec - loc
        occ -= 1
        loc -= diff / occ
        cov -= (np.outer(diff, x_vec - loc) - cov) / np.maximum(occ - self.ddof, 1)

        # numpy -> dict
        self._set_stats(x.keys(), occ, loc, cov)

        return self

    def update_many(self, X: pd.DataFrame):
        """Update with a dataframe of samples.

        Parameters
        ----------
        X
            A dataframe of samples.

        """

        # dict -> numpy
        X_vec = X.values
        occ, loc, cov = self._get_stats(variables=X.columns)

        # update formulas
        diff = X_vec - loc
        m = len(X_vec)
        occ += m
        loc = (occ - m) / occ * loc + m / occ * X_vec.mean(axis=0)
        cov += (diff.T @ (X_vec - loc) - m * cov) / np.maximum(occ - self.ddof, 1)

        # numpy -> dict
        self._set_stats(X.columns, occ, loc, cov)

        return self

    def _get_stats(self, variables):
        """

        Loads means and covariances stored as dictionaries into numpy arrays.

        """
        occ = np.array([self._occ.get(feature, 0.) for feature in variables])
        loc = np.array([self._loc.get(feature, 0.) for feature in variables])
        cov = np.array([
            [self._cov.get((i, j), 0.) for j in variables]
            for i in variables
        ])
        return occ, loc, cov

    def _set_stats(self, variables, occ, loc, cov):
        """

        Takes numpy arrays and stores them as dictionaries with features as keys.

        """
        for i, fi in enumerate(variables):
            self._occ[fi] = occ[i]
            self._loc[fi] = loc[i]
            row = cov[i]
            for j, fj in enumerate(variables):
                self._cov[fi, fj] = row[j]



def _sherman_morrison_inplace(A, u, v):
    """

    From https://timvieira.github.io/blog/post/2021/03/25/fast-rank-one-updates-to-matrix-inverse/

    """
    Au = A @ u
    alpha = -1 / (1 + v.T @ Au)
    sp.linalg.blas.dger(alpha, Au, v.T @ A, a=A, overwrite_a=1)


class EmpiricalPrecision(SymmetricMatrix):
    """Empirical precision matrix.

    The precision matrix is the inverse of the covariance matrix.

    Parameters
    ----------
    ddof
        Delta Degrees of Freedom.

    Examples
    --------

    >>> import numpy as np
    >>> import pandas as pd
    >>> from river import covariance

    >>> np.random.seed(42)
    >>> X = pd.DataFrame(np.random.random((8, 3)), columns=["red", "green", "blue"])
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

    >>> prec = covariance.EmpiricalPrecision()
    >>> for x in X.to_dict(orient="records"):
    ...     prec = prec.update(x)
    >>> prec
            blue     green    red
     blue    5.262   -0.387   0.147
    green   -0.387    4.732   1.119
      red    0.147    1.119   5.428

    There is also an `update_many` method to process mini-batches. The results are identical.

    References
    ----------
    [^1]: [Online Estimation of the Inverse Covariance Matrix - Markus Thill](https://markusthill.github.io/math/stats/ml/online-estimation-of-the-inverse-covariance-matrix/)
    [^2]: [Fast rank-one updates to matrix inverse? - Tim Vieira](https://timvieira.github.io/blog/post/2021/03/25/fast-rank-one-updates-to-matrix-inverse/)

    """

    def __init__(self, ddof=1):
        self.ddof = ddof
        self._w = 0
        self._loc = {}
        self._inv_cov = {}

    @property
    def matrix(self):
        return self._inv_cov

    def update(self, x):
        """Update with a single sample.

        Parameters
        ----------
        x
            A sample.

        """

        # dict -> numpy
        x_vec = np.array(list(x.values()))
        loc = np.array([self._loc.get(feature, 0.) for feature in x])
        # Fortran order is necessary for scipy's linalg.blas.dger
        inv_cov = np.array([
            [self._inv_cov.get((i, j), 1. if i == j else 0.) for j in x]
            for i in x
        ], order='F') / max(self._w, 1)

        # update formulas
        self._w += 1
        diff = x_vec - loc
        loc += diff / self._w
        _sherman_morrison_inplace(A=inv_cov, u=diff, v=x_vec - loc)

        # numpy -> dict
        for i, fi in enumerate(x):
            self._loc[fi] = loc[i]
            row = self._w * inv_cov[i]
            for j, fj in enumerate(x):
                self._inv_cov[fi, fj] = row[j]

        return self
