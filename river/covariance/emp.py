from __future__ import annotations

import abc
import itertools

import numpy as np
import pandas as pd

from river import stats, utils


class SymmetricMatrix(abc.ABC):
    _fmt = ",.3f"

    @property
    @abc.abstractmethod
    def matrix(self):
        ...

    def __getitem__(self, key):
        """

        A covariance matrix is symmetric. For ease of use we make the __getitem__ method symmetric.

        """
        i, j = key
        try:
            return self.matrix[i, j]
        except KeyError:
            return self.matrix[j, i]

    def __repr__(self):
        names = sorted({i for i, _ in self.matrix})

        headers = [""] + list(map(str, names))
        columns = [headers[1:]]
        for col in names:
            column = []
            for row in names:
                try:
                    val = (
                        self[row, col].get()
                        if isinstance(self[row, col], stats.base.Statistic)
                        else self[row, col]
                    )
                    column.append(f"{val:{self._fmt}}")
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
    Cov: 0.020292

    Diagonal entries are variances:

    >>> cov["blue", "blue"]
    Var: 0.076119

    """

    def __init__(self, ddof=1):
        self.ddof = ddof
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

        for i, j in itertools.combinations(sorted(x), r=2):
            try:
                cov = self[i, j]
            except KeyError:
                self._cov[i, j] = stats.Cov(self.ddof)
                cov = self[i, j]
            cov.update(x[i], x[j])

        for i, xi in x.items():
            try:
                var = self[i, i]
            except KeyError:
                self._cov[i, i] = stats.Var(self.ddof)
                var = self[i, i]
            var.update(xi)

        return self

    def revert(self, x: dict):
        """Downdate with a single sample.

        Parameters
        ----------
        x
            A sample.

        """

        for i, j in itertools.combinations(sorted(x), r=2):
            self[i, j].revert(x[i], x[j])

        for i, xi in x.items():
            self[i, i].revert(x[i])

        return self

    def update_many(self, X: pd.DataFrame):
        """Update with a dataframe of samples.

        Parameters
        ----------
        X
            A dataframe of samples.

        """

        X_arr = X.values
        mean_arr = X_arr.mean(axis=0)
        cov_arr = np.cov(X_arr.T, ddof=self.ddof)

        mean = dict(zip(X.columns, mean_arr))
        cov = {
            (i, j): cov_arr[r, c]
            for (r, i), (c, j) in itertools.combinations_with_replacement(enumerate(X.columns), r=2)
        }

        for i, j in itertools.combinations(sorted(X.columns), r=2):
            try:
                self[i, j]
            except KeyError:
                self._cov[i, j] = stats.Cov(self.ddof)
            self._cov[i, j] += stats.Cov._from_state(
                n=len(X),
                mean_x=mean[i],
                mean_y=mean[j],
                cov=cov.get((i, j), cov.get((j, i))),
                ddof=self.ddof,
            )

        for i in X.columns:
            try:
                self[i, i]
            except KeyError:
                self._cov[i, i] = stats.Var(self.ddof)
            self._cov[i, i] += stats.Var._from_state(
                n=len(X), m=mean[i], sig=cov[i, i], ddof=self.ddof
            )

        return self


class EmpiricalPrecision(SymmetricMatrix):
    """Empirical precision matrix.

    The precision matrix is the inverse of the covariance matrix.

    This implementation leverages the Sherman-Morrison formula. The resulting inverse covariance
    matrix is not guaranteed to be identical to a batch computation. However, the difference
    shrinks with the number of observations.

    Examples
    --------

    >>> import numpy as np
    >>> import pandas as pd
    >>> from river import covariance

    >>> np.random.seed(42)
    >>> X = pd.DataFrame(np.random.random((1000, 3)))
    >>> X.head()
              0         1         2
    0  0.374540  0.950714  0.731994
    1  0.598658  0.156019  0.155995
    2  0.058084  0.866176  0.601115
    3  0.708073  0.020584  0.969910
    4  0.832443  0.212339  0.181825

    >>> prec = covariance.EmpiricalPrecision()
    >>> for x in X.to_dict(orient="records"):
    ...     prec = prec.update(x)

    >>> prec
        0        1        2
    0   12.026   -0.122   -0.214
    1   -0.122   11.276   -0.026
    2   -0.214   -0.026   11.632

    >>> pd.DataFrame(np.linalg.inv(np.cov(X.T, ddof=1)))
               0          1          2
    0  12.159791  -0.124966  -0.218671
    1  -0.124966  11.393394  -0.026662
    2  -0.218671  -0.026662  11.756907

    References
    ----------
    [^1]: [Online Estimation of the Inverse Covariance Matrix - Markus Thill](https://markusthill.github.io/math/stats/ml/online-estimation-of-the-inverse-covariance-matrix/)
    [^2]: [Fast rank-one updates to matrix inverse? - Tim Vieira](https://timvieira.github.io/blog/post/2021/03/25/fast-rank-one-updates-to-matrix-inverse/)
    [^3]: [Woodbury matrix identity](https://www.wikiwand.com/en/Woodbury_matrix_identity)

    """

    def __init__(self):
        self._w = {}
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
        loc = np.array([self._loc.get(feature, 0.0) for feature in x])
        w = np.array([self._w.get(feature, 0.0) for feature in x])
        # Fortran order is necessary for scipy's linalg.blas.dger
        inv_cov = np.array(
            [
                [self._inv_cov.get(min((i, j), (j, i)), 1.0 if i == j else 0.0) for j in x]
                for i in x
            ],
            order="F",
        ) / np.maximum(w, 1)

        # update formulas
        w += 1
        diff = x_vec - loc
        loc += diff / w
        utils.math.sherman_morrison(A=inv_cov, u=diff, v=x_vec - loc)

        # numpy -> dict
        for i, fi in enumerate(x):
            self._loc[fi] = loc[i]
            self._w[fi] = w[i]
            row = self._w[fi] * inv_cov[i]
            for j, fj in enumerate(x):
                self._inv_cov[min((fi, fj), (fj, fi))] = row[j]

        return self

    def update_many(self, X: pd.DataFrame):
        """Update with a dataframe of samples.

        Parameters
        ----------
        X
            A dataframe of samples.

        """

        # numpy -> dict
        X_arr = X.values
        loc = np.array([self._loc.get(feature, 0.0) for feature in X])
        w = np.array([self._w.get(feature, 0.0) for feature in X])
        inv_cov = np.array(
            [[self._inv_cov.get(min((i, j), (j, i)), 1.0 if i == j else 0.0) for j in X] for i in X]
        ) / np.maximum(w, 1)

        # update formulas
        diff = X_arr - loc
        loc = (w * loc + len(X) * X_arr.mean(axis=0)) / (w + len(X))
        w += len(X)
        utils.math.woodbury_matrix(A=inv_cov, U=diff.T, V=X_arr - loc)

        # numpy -> dict
        for i, fi in enumerate(X):
            self._loc[fi] = loc[i]
            self._w[fi] = w[i]
            row = self._w[fi] * inv_cov[i]
            for j, fj in enumerate(X):
                self._inv_cov[min((fi, fj), (fj, fi))] = row[j]

        return self
