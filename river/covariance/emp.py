from __future__ import annotations

import abc
import itertools
import typing

import numpy as np

from river import stats, utils

if typing.TYPE_CHECKING:
    from narwhals.stable.v2.typing import IntoDataFrame


class SymmetricMatrix(abc.ABC):
    _fmt = ",.3f"

    @property
    @abc.abstractmethod
    def matrix(self) -> dict: ...

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
        if not names:
            return f"{type(self).__name__} (empty)"

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
    ...     cov.update(x)
    >>> cov
            blue     green    red
     blue    0.076    0.020   -0.010
    green    0.020    0.113   -0.053
      red   -0.010   -0.053    0.079

    There is also an `update_many` method to process mini-batches. The results are identical.

    >>> cov = covariance.EmpiricalCovariance()
    >>> cov.update_many(X)
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
        self._cached_keys: tuple = ()
        self._cached_pairs: list[tuple] = []

    @property
    def matrix(self):
        return self._cov

    def _pairs_for(self, x: dict):
        keys = tuple(sorted(x))
        if keys != self._cached_keys:
            self._cached_keys = keys
            self._cached_pairs = list(itertools.combinations(keys, 2))
        return self._cached_keys, self._cached_pairs

    def update(self, x: dict, w: float = 1.0):
        """Update with a single sample.

        Parameters
        ----------
        x
            A sample.

        """
        ddof = self.ddof
        cov_dict = self._cov
        keys, pairs = self._pairs_for(x)

        for key in pairs:
            cov = cov_dict.get(key)
            if cov is None:
                cov = stats.Cov(ddof)
                cov_dict[key] = cov
            i, j = key
            cov.update(x[i], x[j], w)

        for i in keys:
            key = (i, i)
            var = cov_dict.get(key)
            if var is None:
                var = stats.Var(ddof)
                cov_dict[key] = var
            var.update(x[i], w)

    def revert(self, x: dict, w: float = 1.0):
        """Downdate with a single sample.

        Parameters
        ----------
        x
            A sample.

        """
        cov_dict = self._cov
        keys, pairs = self._pairs_for(x)

        for key in pairs:
            i, j = key
            cov_dict[key].revert(x[i], x[j], w)

        for i in keys:
            cov_dict[i, i].revert(x[i], w)

    def update_many(self, X: IntoDataFrame):
        """Update with a dataframe of samples.

        Any [narwhals](https://github.com/narwhals-dev/narwhals)-compatible eager dataframe
        (pandas, polars, pyarrow, ...) is accepted.

        Parameters
        ----------
        X
            A dataframe of samples.

        """

        frame = utils.dataframe.into_frame(X)
        columns = list(frame.columns)
        X_arr = utils.dataframe.to_numpy(frame)
        mean_arr = X_arr.mean(axis=0)
        cov_arr = np.cov(X_arr.T, ddof=self.ddof)

        n = len(frame)
        mean = dict(zip(columns, mean_arr))
        cov = {
            (i, j): cov_arr[r, c]
            for (r, i), (c, j) in itertools.combinations_with_replacement(enumerate(columns), r=2)
        }

        self._update_from_state(n=n, mean=mean, cov=cov)

    def _update_from_state(self, n: int, mean: dict, cov: float | dict):
        """Update from state information.

        Parameters
        ----------
        n
            The number of data points.
        mean
            A dictionary of variable means.
        cov
            A dictionary of covariance or variance values.

        Raises
        ----------
            KeyError: If an element in `mean` or `cov` is missing.

        """
        for i, j in itertools.combinations(sorted(mean.keys()), r=2):
            try:
                self[i, j]
            except KeyError:
                self._cov[i, j] = stats.Cov(self.ddof)
            if isinstance(cov, dict):
                cov_ = cov.get((i, j), cov.get((j, i)))
            else:
                cov_ = cov
            self._cov[i, j] += stats.Cov._from_state(
                n=n,
                mean_x=mean[i],
                mean_y=mean[j],
                cov=cov_,
                ddof=self.ddof,
            )

        for i in mean.keys():
            try:
                self[i, i]
            except KeyError:
                self._cov[i, i] = stats.Var(self.ddof)
            if isinstance(cov, dict):
                cov_ = cov[i, i]
            else:
                cov_ = cov
            self._cov[i, i] += stats.Var._from_state(n=n, m=mean[i], sig=cov_, ddof=self.ddof)

    @classmethod
    def _from_state(cls, n: int, mean: dict, cov: float | dict, *, ddof=1):
        """Create a new instance from state information.

        Parameters
        ----------
        cls
            The class type.
        n
            The number of data points.
        mean
            A dictionary of variable means.
        cov
            A dictionary of covariance or variance values.
        ddof
            Degrees of freedom for covariance calculation. Defaults to 1.

        Returns
        ----------
            cls: A new instance of the class with updated covariance matrix.

        """
        new = cls(ddof=ddof)
        new._update_from_state(n=n, mean=mean, cov=cov)
        return new


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
    ...     prec.update(x)

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
        self._idx: dict = {}
        self._loc_arr = np.zeros(0, dtype=np.float64)
        self._w_arr = np.zeros(0, dtype=np.float64)
        self._inv_cov_mat = np.zeros((0, 0), dtype=np.float64, order="F")
        self._cap = 0

    def _grow(self, needed: int) -> None:
        new_cap = max(needed, max(8, self._cap * 2))
        new_loc = np.zeros(new_cap, dtype=np.float64)
        new_w = np.zeros(new_cap, dtype=np.float64)
        new_inv = np.eye(new_cap, dtype=np.float64, order="F")
        if self._cap:
            new_loc[: self._cap] = self._loc_arr
            new_w[: self._cap] = self._w_arr
            new_inv[: self._cap, : self._cap] = self._inv_cov_mat
        self._loc_arr = new_loc
        self._w_arr = new_w
        self._inv_cov_mat = new_inv
        self._cap = new_cap

    def _ensure_features(self, features) -> np.ndarray:
        idx = self._idx
        ids = []
        for f in features:
            i = idx.get(f)
            if i is None:
                i = len(idx)
                idx[f] = i
            ids.append(i)
        if len(idx) > self._cap:
            self._grow(len(idx))
        return np.asarray(ids, dtype=np.intp)

    @property
    def matrix(self) -> dict:
        mat = self._inv_cov_mat
        features = list(self._idx)
        out = {}
        for ai, fa in enumerate(features):
            for bi in range(ai, len(features)):
                fb = features[bi]
                out[min((fa, fb), (fb, fa))] = mat[ai, bi]
        return out

    def __getitem__(self, key):
        i, j = key
        ai = self._idx.get(i)
        bi = self._idx.get(j)
        if ai is None or bi is None:
            raise KeyError(key)
        return self._inv_cov_mat[ai, bi]

    def update(self, x):
        """Update with a single sample.

        Parameters
        ----------
        x
            A sample.

        """
        ids = self._ensure_features(x.keys())
        x_vec = np.fromiter(x.values(), dtype=np.float64, count=len(x))

        loc = self._loc_arr[ids].copy()
        w = self._w_arr[ids].copy()
        # Fortran order is necessary for scipy's linalg.blas.dger
        ix = np.ix_(ids, ids)
        inv_cov = np.asfortranarray(self._inv_cov_mat[ix]) / np.maximum(w, 1)

        # update formulas
        w += 1
        diff = x_vec - loc
        loc += diff / w
        utils.math.sherman_morrison(A=inv_cov, u=diff, v=x_vec - loc)

        # scatter back to dense state — symmetrize so [a, b] == [b, a],
        # which matters when features arrive at different times (the
        # per-feature `w` scaling would otherwise leave the matrix skewed).
        block = w[:, None] * inv_cov
        self._loc_arr[ids] = loc
        self._w_arr[ids] = w
        self._inv_cov_mat[ix] = 0.5 * (block + block.T)

    def update_many(self, X: IntoDataFrame):
        """Update with a dataframe of samples.

        Any [narwhals](https://github.com/narwhals-dev/narwhals)-compatible eager dataframe
        (pandas, polars, pyarrow, ...) is accepted.

        Parameters
        ----------
        X
            A dataframe of samples.

        """
        frame = utils.dataframe.into_frame(X)
        ids = self._ensure_features(frame.columns)
        X_arr = utils.dataframe.to_numpy(frame)

        loc = self._loc_arr[ids].copy()
        w = self._w_arr[ids].copy()
        ix = np.ix_(ids, ids)
        inv_cov = np.asfortranarray(self._inv_cov_mat[ix]) / np.maximum(w, 1)

        # update formulas
        n_batch = len(frame)
        diff = X_arr - loc
        loc = (w * loc + n_batch * X_arr.mean(axis=0)) / (w + n_batch)
        w += n_batch
        utils.math.woodbury_matrix(A=inv_cov, U=diff.T, V=X_arr - loc)

        # scatter back to dense state (see `update` for why we symmetrize)
        block = w[:, None] * inv_cov
        self._loc_arr[ids] = loc
        self._w_arr[ids] = w
        self._inv_cov_mat[ix] = 0.5 * (block + block.T)
