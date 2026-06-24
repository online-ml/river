from __future__ import annotations

import typing

import numpy as np

from river import base, proba, utils

if typing.TYPE_CHECKING:
    from narwhals.stable.v2.typing import IntoDataFrame, IntoSeries


class BayesianLinearRegression(base.MiniBatchRegressor):
    """Bayesian linear regression.

    An advantage of Bayesian linear regression over standard linear regression is that features
    do not have to scaled beforehand. Another attractive property is that this flavor of linear
    regression is somewhat insensitive to its hyperparameters. Finally, this model can output
    instead a predictive distribution rather than just a point estimate.

    The downside is that the learning step runs in `O(n^2)` time, whereas the learning step of
    standard linear regression takes `O(n)` time.

    Parameters
    ----------
    alpha
        Prior parameter.
    beta
        Noise parameter.
    smoothing
        Smoothing allows the model to gradually "forget" the past, and focus on the more recent
        data. It thus enables the model to deal with concept drift. Due to the current
        implementation, activating smoothing may slow down the model.

    Examples
    --------

    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import linear_model
    >>> from river import metrics

    >>> dataset = datasets.TrumpApproval()
    >>> model = linear_model.BayesianLinearRegression()
    >>> metric = metrics.MAE()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    MAE: 0.586...

    >>> x, _ = next(iter(dataset))
    >>> model.predict_one(x)
    43.855...

    >>> model.predict_one(x, with_dist=True)
    𝒩(μ=43.85..., σ=1.00...)

    The `smoothing` parameter can be set to make the model robust to drift. The parameter is
    expected to be between 0 and 1. To exemplify, let's generate some simulation data with an
    abrupt concept drift right in the middle.

    >>> import itertools
    >>> import random

    >>> def random_data(coefs, n, seed=42):
    ...     rng = random.Random(seed)
    ...     for _ in range(n):
    ...         x = {i: rng.random() for i, c in enumerate(coefs)}
    ...         y = sum(c * xi for c, xi in zip(coefs, x.values()))
    ...         yield x, y

    Here's how the model performs without any smoothing:

    >>> model = linear_model.BayesianLinearRegression()
    >>> dataset = itertools.chain(
    ...     random_data([0.1, 3], 100),
    ...     random_data([10, -2], 100)
    ... )
    >>> metric = metrics.MAE()
    >>> evaluate.progressive_val_score(dataset, model, metric)
    MAE: 1.284...

    And here's how it performs with some smoothing:

    >>> model = linear_model.BayesianLinearRegression(smoothing=0.8)
    >>> dataset = itertools.chain(
    ...     random_data([0.1, 3], 100),
    ...     random_data([10, -2], 100)
    ... )
    >>> metric = metrics.MAE()
    >>> evaluate.progressive_val_score(dataset, model, metric)
    MAE: 0.159...

    Smoothing allows the model to gradually "forget" the past, and focus on the more recent data.

    Note how this works better than standard linear regression, even when using an aggressive
    learning rate.

    >>> from river import optim
    >>> model = linear_model.LinearRegression(optimizer=optim.SGD(0.5))
    >>> dataset = itertools.chain(
    ...     random_data([0.1, 3], 100),
    ...     random_data([10, -2], 100)
    ... )
    >>> metric = metrics.MAE()
    >>> evaluate.progressive_val_score(dataset, model, metric)
    MAE: 0.242...

    Features that are absent from `x` in `learn_one` are treated as observed values of 0. This is
    the right default when features are centered around zero (e.g. sparse counts or indicators),
    but biases the posterior when typical feature values are far from zero. In that case, combine
    the model with `preprocessing.StatImputer` in a pipeline so that missing observations are
    filled with a running statistic before they reach the model:

    >>> from river import preprocessing
    >>> from river import stats

    >>> def with_missing(dataset, p_missing=0.2, seed=42):
    ...     rng = random.Random(seed)
    ...     for x, y in dataset:
    ...         x = {f: (None if rng.random() < p_missing else v) for f, v in x.items()}
    ...         yield x, y

    >>> features = ['ordinal_date', 'gallup', 'ipsos', 'morning_consult',
    ...             'rasmussen', 'you_gov']
    >>> model = (
    ...     preprocessing.StatImputer(*[(f, stats.Mean()) for f in features])
    ...     | linear_model.BayesianLinearRegression()
    ... )
    >>> metric = metrics.MAE()
    >>> evaluate.progressive_val_score(
    ...     with_missing(datasets.TrumpApproval()), model, metric
    ... )
    MAE: 0.774...

    References
    ----------
    [^1]: [Pattern Recognition and Machine Learning, page 52 — Christopher M. Bishop](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)
    [^2]: [Bayesian/Streaming Algorithms — Vincent Warmerdam](https://koaning.io/posts/bayesian-propto-streaming/)
    [^3]: [Bayesian linear regression for practitioners — Max Halford](https://maxhalford.github.io/blog/bayesian-linear-regression/)

    """

    def __init__(self, alpha=1, beta=1, smoothing: float | None = None):
        self.alpha = alpha
        self.beta = beta
        self.smoothing = smoothing
        self._idx: dict[base.typing.FeatureName, int] = {}
        # We track:
        # - `_ss_arr`: the posterior precision matrix, updated by exact rank-1
        #   additions and so identical across BLAS implementations.
        # - `_ss_inv_arr`: the posterior covariance, maintained in-place via a
        #   Sherman-Morrison rank-1 update (one BLAS dger per `learn_one`).
        # - `_eta_arr = beta * sum_i(y_i * x_i)`: the natural mean, also an
        #   exact accumulator.
        # The posterior mean is recovered from these on demand as
        # `_ss_inv_arr @ _eta_arr` and cached in `_m_arr` until the next
        # `learn_one`. Recomputing `m` from the natural parameters rather than
        # propagating it through `m_new = ss_inv @ (ss_old @ m_old + bx*y)`
        # keeps cross-platform drift roughly linear in BLAS rounding rather
        # than compounding multiplicatively.
        self._ss_arr = np.zeros((0, 0), dtype=np.float64)
        self._ss_inv_arr = np.zeros((0, 0), dtype=np.float64, order="F")
        self._eta_arr = np.zeros(0, dtype=np.float64)
        self._m_arr = np.zeros(0, dtype=np.float64)
        self._m_dirty = False
        self._cap = 0
        self._n = 1

    def _grow(self, needed: int) -> None:
        new_cap = max(needed, max(8, self._cap * 2))
        diag = 1.0 / self.alpha
        new_ss = np.eye(new_cap, dtype=np.float64) * diag
        new_ss_inv = np.eye(new_cap, dtype=np.float64, order="F") * diag
        new_eta = np.zeros(new_cap, dtype=np.float64)
        if self._cap:
            new_ss[: self._cap, : self._cap] = self._ss_arr
            new_ss_inv[: self._cap, : self._cap] = self._ss_inv_arr
            new_eta[: self._cap] = self._eta_arr
        self._ss_arr = new_ss
        self._ss_inv_arr = new_ss_inv
        self._eta_arr = new_eta
        self._cap = new_cap

    def _ensure_features(self, features) -> None:
        idx = self._idx
        for f in features:
            if f not in idx:
                idx[f] = len(idx)
        if len(idx) > self._cap:
            self._grow(len(idx))

    def _ensure_m(self) -> None:
        if not self._m_dirty:
            return
        if self._idx:
            self._m_arr = self._ss_inv_arr @ self._eta_arr
        self._m_dirty = False

    def learn_one(self, x, y):
        # Treat features absent from `x` as observed values of 0. Updating the
        # full-cap state (rather than just the touched submatrix via np.ix_)
        # keeps `_ss_inv_arr = inv(_ss_arr)` consistent across
        # emerging/disappearing feature subsets.
        self._ensure_features(x.keys())
        x_arr = np.zeros(self._cap, dtype=np.float64)
        for f, v in x.items():
            x_arr[self._idx[f]] = v

        bx = self.beta * x_arr

        if self.smoothing is None:
            # Bishop equation 3.51
            self._ss_arr += np.outer(bx, x_arr)
            utils.math.sherman_morrison(A=self._ss_inv_arr, u=bx, v=x_arr)
            # Natural mean: eta_N = eta_{N-1} + beta * y_N * x_N
            self._eta_arr += bx * y
        else:
            s = self.smoothing
            new_ss_arr = s * self._ss_arr + (1 - s) * np.outer(bx, x_arr)
            # TODO: we use standard matrix inversion. This is not very efficient. However, we don't
            # yet have a formula for the Sherman-Morrison approximation when a smoothing factor is
            # involved. This is an interesting research direction!
            self._ss_inv_arr = np.asfortranarray(np.linalg.inv(new_ss_arr))
            self._ss_arr = new_ss_arr
            self._eta_arr = s * self._eta_arr + (1 - s) * bx * y

        self._m_dirty = True

    def learn_many(self, X: IntoDataFrame, y: IntoSeries) -> None:
        # A mini-batch is exactly the sequence of rank-1 `learn_one` updates applied in row
        # order, collapsed into a handful of matrix products. narwhals stays at the boundary:
        # the input is wrapped, then a float64 matrix + column names drive the numpy core.
        Xnw = utils.dataframe.into_frame(X)
        cols = Xnw.columns
        self._ensure_features(cols)
        idx = np.fromiter((self._idx[f] for f in cols), dtype=np.intp, count=len(cols))

        X_arr = utils.dataframe.to_numpy(Xnw)  # (n_samples, n_features)
        y_arr = np.asarray(utils.dataframe.into_series(y).to_numpy(), dtype=np.float64)
        n = len(self._idx)

        if self.smoothing is None:
            # Bishop equations 3.50/3.51 in batch form. Summing the per-row rank-1 outer
            # products into a single Gram matrix is identical to the `learn_one` loop:
            #   precision    += beta * sum_i x_i x_i^T = beta * X^T X
            #   natural mean += beta * sum_i y_i x_i   = beta * X^T y
            self._ss_arr[np.ix_(idx, idx)] += self.beta * (X_arr.T @ X_arr)
            self._eta_arr[idx] += self.beta * (X_arr.T @ y_arr)
        else:
            # With smoothing each row applies an EMA step `S <- s*S + (1-s)*beta*x x^T`, so
            # after N rows (oldest first) the prior is decayed by s^N and row k (0-based)
            # carries weight (1-s)*s^(N-1-k). This is the exact closed form of the loop.
            s = self.smoothing
            n_samples = len(X_arr)
            weights = (1 - s) * s ** np.arange(n_samples - 1, -1, -1)
            decay = s**n_samples
            self._ss_arr[:n, :n] *= decay
            self._eta_arr[:n] *= decay
            self._ss_arr[np.ix_(idx, idx)] += self.beta * (X_arr.T * weights) @ X_arr
            self._eta_arr[idx] += self.beta * (X_arr.T @ (weights * y_arr))

        # `learn_one` keeps `_ss_inv_arr = inv(_ss_arr)` (via Sherman-Morrison without smoothing,
        # a full inverse with it). Refresh the active block in one inverse so subsequent
        # `learn_one` updates and `predict_one(..., with_dist=True)` see a consistent covariance.
        if n:
            self._ss_inv_arr[:n, :n] = np.linalg.inv(self._ss_arr[:n, :n])
        self._m_dirty = True

    def predict_one(self, x, with_dist=False):
        """Predict the output of features `x`.

        Parameters
        ----------
        x
            A dictionary of features.
        with_dist
            Whether to return a predictive distribution, or instead just the most likely value.

        Returns
        -------
        The prediction.

        """

        # Bishop equation 3.58
        y_pred_mean = 0.0
        if self._idx:
            self._ensure_m()
            m_arr = self._m_arr
            for f, v in x.items():
                i = self._idx.get(f)
                if i is not None:
                    y_pred_mean += m_arr[i] * v

        if not with_dist:
            return float(y_pred_mean)

        n = len(self._idx)
        diag = 1.0 / self.alpha
        x_arr = np.zeros(self._cap, dtype=np.float64)
        unknown_norm_sq = 0.0
        for f, v in x.items():
            i = self._idx.get(f)
            if i is None:
                unknown_norm_sq += float(v) * float(v)
            elif n:
                x_arr[i] = v
        # Bishop equation 3.59
        if n:
            y_pred_var = 1 / self.beta + float(x_arr @ self._ss_inv_arr @ x_arr)
        else:
            y_pred_var = 1 / self.beta
        y_pred_var += diag * unknown_norm_sq

        return proba.Gaussian._from_state(n=1, m=y_pred_mean, sig=y_pred_var**0.5, ddof=0)

    def predict_many(self, X: IntoDataFrame) -> IntoSeries:
        X = utils.dataframe.into_frame(X)
        self._ensure_m()
        m_full = self._m_arr
        m = np.zeros(len(X.columns), dtype=np.float64)
        for k, f in enumerate(X.columns):
            i = self._idx.get(f)
            if i is not None:
                m[k] = m_full[i]
        y_pred = utils.dataframe.to_numpy(X) @ m
        return utils.dataframe.to_native_series(y_pred, name=None, like=X)
