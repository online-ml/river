import collections
import itertools

import pandas as pd

from river import stats, utils


class CovMatrix(collections.UserDict):
    """Sample covariance matrix.

    Parameters
    ----------
    ddof
        Delta Degrees of Freedom.

    Examples
    --------

    >>> import numpy as np
    >>> import pandas as pd
    >>> from river import misc

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

    >>> cov = misc.CovMatrix()
    >>> for x in X.to_dict(orient="records"):
    ...     cov = cov.update(x)
    >>> cov
            blue     green    red
     blue    0.076    0.020   -0.010
    green    0.020    0.113   -0.053
      red   -0.010   -0.053    0.079

    There is also an `update_many` method to process mini-batches. The results are identical.

    >>> cov = misc.CovMatrix()
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

    _fmt = ",.3f"

    def __init__(self, ddof=1):
        super().__init__()
        self.ddof = ddof

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
                self[i, j] = stats.Cov(self.ddof)
                cov = self[i, j]
            cov.update(x[i], x[j])

        for i, xi in x.items():
            try:
                var = self[i, i]
            except KeyError:
                self[i, i] = stats.Var(self.ddof)
                var = self[i, i]
            var.update(xi)

        return self

    def update_many(self, X: pd.DataFrame):
        """Update with many samples.

        Parameters
        ----------
        X
            Samples.

        """

        for i, j in itertools.combinations(sorted(X.columns), r=2):
            try:
                cov = self[i, j]
            except KeyError:
                self[i, j] = stats.Cov(self.ddof)
                cov = self[i, j]
            cov.update_many(X[i].values, X[j].values)

        for i in X.columns:
            try:
                var = self[i, i]
            except KeyError:
                self[i, i] = stats.Var(self.ddof)
                var = self[i, i]
            var.update_many(X[i].values)

        return self

    def __getitem__(self, key):
        """

        A covariance matrix is symmetric. For ease of use we make the __getitem__ method symmetric.

        """
        x, y = key
        try:
            return super().__getitem__((x, y))
        except KeyError:
            return super().__getitem__((y, x))

    def __repr__(self):

        names = sorted(set(i for i, _ in self))

        headers = [""] + list(map(str, names))
        columns = [headers[1:]]
        for col in names:
            column = []
            for row in names:
                try:
                    column.append(f"{self[row, col].get():{self._fmt}}")
                except KeyError:
                    column.append("")
            columns.append(column)

        return utils.pretty.print_table(headers, columns)
