import itertools

import numpy as np
import pandas as pd

from river import stats, utils


class EmpiricalCovariance:
    """Empirical covariance matrix.

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
        self.ddof = ddof
        self._w = 0
        self._loc = {}
        self._cov = {}

    def update(self, x: dict):
        """Update with a single sample.

        Parameters
        ----------
        x
            A sample.

        """

        # dict -> numpy
        x_vec = np.array(list(x.values()))
        loc = np.array([self._loc.get(feature, 0.) for feature in x])
        cov = np.array([
            [self._cov.get((i, j), 0.) for j in x]
            for i in x
        ])

        # update formulas
        self._w += 1
        d = x_vec - loc
        loc += d / self._w
        cov += (np.outer(d, x_vec - loc) - cov) / max(self._w - self.ddof, 1)

        # numpy -> dict
        for i, fi in enumerate(x):
            self._loc[fi] = loc[i]
            row = cov[i]
            for j, fj in enumerate(x):
                self._cov[fi, fj] = row[j]

        return self

    def update_many(self, X: pd.DataFrame):
        """Update with many samples.

        Parameters
        ----------
        X
            Samples.

        """
        raise NotImplementedError


    def __getitem__(self, key):
        """

        A covariance matrix is symmetric. For ease of use we make the __getitem__ method symmetric.

        """
        x, y = key
        try:
            return self._cov[x, y]
        except KeyError:
            return self._cov[y, x]

    def __repr__(self):

        names = sorted(set(i for i, _ in self._cov))

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




class EmpiricalPrecision:
    """Empirical precision matrix.

    The precision matrix is the inverse of the covariance matrix.

    References

    ----------
    [^1]: [Online Estimation of the Inverse Covariance Matrix - Markus Thill](https://markusthill.github.io/math/stats/ml/online-estimation-of-the-inverse-covariance-matrix/)
    [^2]: [Fast rank-one updates to matrix inverse? - Tim Vieira](https://timvieira.github.io/blog/post/2021/03/25/fast-rank-one-updates-to-matrix-inverse/)

    """
