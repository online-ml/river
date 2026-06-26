from __future__ import annotations

import collections
import functools
import math
import typing

import narwhals as nw
import numpy as np

from river import proba
from river.utils.dataframe import into_frame, to_native_frame

from . import base

if typing.TYPE_CHECKING:
    from narwhals.stable.v2.typing import IntoDataFrame, IntoSeries

__all__ = ["GaussianNB"]

_PDF_EPS = 10e-10
_LOG_PDF_EPS = math.log(_PDF_EPS)


class GaussianNB(base.BaseNB):
    """Gaussian Naive Bayes.

    A Gaussian distribution $G_{cf}$ is maintained for each class $c$ and each feature $f$. Each
    Gaussian is updated using the amount associated with each feature; the details can be be found
    in `proba.Gaussian`. The joint log-likelihood is then obtained by summing the log probabilities
    of each feature associated with each class.

    Examples
    --------

    >>> from river import naive_bayes
    >>> from river import stream
    >>> import numpy as np

    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> Y = np.array([1, 1, 1, 2, 2, 2])

    >>> model = naive_bayes.GaussianNB()

    >>> for x, y in stream.iter_array(X, Y):
    ...     model.learn_one(x, y)

    >>> model.predict_one({0: -0.8, 1: -1})
    1

    You can also train the model and make predictions in mini-batch mode.

    >>> import pandas as pd

    >>> model = naive_bayes.GaussianNB()
    >>> model.learn_many(pd.DataFrame(X), pd.Series(Y))
    >>> model.predict_many(pd.DataFrame([[-0.8, -1], [2.8, 1.5]]))
    0    1
    1    2
    dtype: int64

    """

    def __init__(self):
        self.class_counts = collections.Counter()
        self.gaussians = collections.defaultdict(
            functools.partial(collections.defaultdict, proba.Gaussian)
        )

    def learn_one(self, x, y):
        self.class_counts.update((y,))

        for i, xi in x.items():
            self.gaussians[y][i].update(xi)

    def p_class(self, c):
        return self.class_counts[c] / sum(self.class_counts.values())

    @staticmethod
    def _log_gaussian_pdf(gaussian: proba.Gaussian | None, x) -> float:
        if gaussian is None:
            return _LOG_PDF_EPS
        return math.log(_PDF_EPS + gaussian(x))

    def joint_log_likelihood(self, x):
        if not self.class_counts:
            return {}

        return {
            c: math.log(self.p_class(c))
            + sum(
                self._log_gaussian_pdf(self.gaussians.get(c, {}).get(i), xi) for i, xi in x.items()
            )
            for c in self.class_counts
        }

    def learn_many(self, X: IntoDataFrame, y: IntoSeries):
        """Learn from a batch of feature vectors.

        Parameters
        ----------
        X
            Feature vectors.
        y
            Target classes.

        """
        if hasattr(X, "sparse"):
            X = X.sparse.to_dense()

        X = nw.from_native(X, eager_only=True)
        y = nw.from_native(y, series_only=True)

        self.class_counts.update(y.to_list())

        for c in y.unique().to_list():
            mask = (y == c).rename("mask")
            X_c = X.filter(mask)

            for i in X_c.columns:
                values = X_c[i].drop_nulls()
                if len(values) == 0:
                    continue
                self.gaussians[c][i]._var.update_many(values.to_numpy().astype(float))

    @staticmethod
    def _log_gaussian_pdf_many(gaussian: proba.Gaussian | None, values: np.ndarray) -> np.ndarray:
        if gaussian is None:
            return np.full(values.shape, _LOG_PDF_EPS)

        var = gaussian._var
        n = var.mean.n
        if n > var.ddof:
            variance = var._S / (n - var.ddof)
            if variance > 0.0:
                mu = var.mean._mean
                with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                    pdf = np.exp((values - mu) ** 2 / (-2.0 * variance)) / math.sqrt(
                        math.tau * variance
                    )
                return np.log(_PDF_EPS + pdf)

        return np.full(values.shape, _LOG_PDF_EPS)

    def joint_log_likelihood_many(self, X: IntoDataFrame) -> IntoDataFrame:
        """Compute the joint log-likelihoods for a batch of feature vectors.

        Parameters
        ----------
        X
            Feature vectors.

        Returns
        -------
        Input samples joint log-likelihoods.

        """

        if hasattr(X, "sparse"):
            X = X.sparse.to_dense()

        X = nw.from_native(X, eager_only=True)

        if not self.class_counts:
            return X.select([]).to_native()

        jll = {}

        for c in self.class_counts:
            ll = np.full(len(X), math.log(self.p_class(c)), dtype=float)
            gaussians = self.gaussians.get(c, {})
    
            for col in X.columns:
                s = X[col].to_numpy().astype(object)
                mask = np.array([v is not None for v in s])
    
                values = np.asarray(s[mask]).astype(float)
    
                col_ll = np.full(len(X), _LOG_PDF_EPS, dtype=float)
                col_ll[mask] = self._log_gaussian_pdf_many(
                    gaussians.get(col),
                    values,
                )
                
                ll += col_ll
    
            jll[c] = ll

    return to_native_frame(jll, like=X)

    def _unit_test_skips(self):
        return set()
