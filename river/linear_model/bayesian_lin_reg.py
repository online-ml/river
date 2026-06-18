from __future__ import annotations

import typing

import numpy as np

from river import base, proba, utils

if typing.TYPE_CHECKING:
    pass


class BayesianLinearRegression(base.Regressor):
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
    43.852...

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
    MAE: 0.775...

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
        self._m_arr = np.zeros(0, dtype=np.float64)
        self._ss_arr = np.zeros((0, 0), dtype=np.float64)
        self._ss_inv_arr = np.zeros((0, 0), dtype=np.float64, order="F")
        self._cap = 0
        self._n = 1

    def _grow(self, needed: int) -> None:
        new_cap = max(needed, max(8, self._cap * 2))
        diag = 1.0 / self.alpha
        new_m = np.zeros(new_cap, dtype=np.float64)
        new_ss = np.eye(new_cap, dtype=np.float64) * diag
        new_ss_inv = np.eye(new_cap, dtype=np.float64, order="F") * diag
        if self._cap:
            new_m[: self._cap] = self._m_arr
            new_ss[: self._cap, : self._cap] = self._ss_arr
            new_ss_inv[: self._cap, : self._cap] = self._ss_inv_arr
        self._m_arr = new_m
        self._ss_arr = new_ss
        self._ss_inv_arr = new_ss_inv
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

    def _gather_ss_inv(self, features) -> np.ndarray:
        """Build the ss_inv submatrix for `features` without mutating state.

        Features not yet in the model get a diagonal of 1/alpha and zeros off-diagonal,
        matching the original dict-default behavior.
        """
        diag = 1.0 / self.alpha
        n = len(features)
        ids = np.fromiter(
            (self._idx.get(f, -1) for f in features),
            dtype=np.intp,
            count=n,
        )
        out = np.eye(n, dtype=np.float64, order="F") * diag
        known_mask = ids >= 0
        if known_mask.any():
            local = np.where(known_mask)[0]
            global_ = ids[local]
            out[np.ix_(local, local)] = self._ss_inv_arr[np.ix_(global_, global_)]
        return out

    def learn_one(self, x, y):
        # Update the full-cap state, treating features absent from `x` as 0.
        # Operating on submatrices via np.ix_(ids, ids) breaks the invariant
        # `_ss_inv_arr = inv(_ss_arr)` once different feature subsets are touched
        # across calls (the submatrix of an inverse ≠ inverse of the submatrix),
        # which causes coefficients to diverge under emerging/disappearing
        # features.
        self._ensure_features(x.keys())
        x_arr = np.zeros(self._cap, dtype=np.float64)
        for f, v in x.items():
            x_arr[self._idx[f]] = v

        bx = self.beta * x_arr
        ss_arr = self._ss_arr
        ss_inv_arr = self._ss_inv_arr

        if self.smoothing is None:
            utils.math.sherman_morrison(A=ss_inv_arr, u=bx, v=x_arr)
            # Bishop equation 3.50
            self._m_arr = ss_inv_arr @ (ss_arr @ self._m_arr + bx * y)
            # Bishop equation 3.51
            ss_arr += np.outer(bx, x_arr)
        else:
            s = self.smoothing
            new_ss_arr = s * ss_arr + (1 - s) * np.outer(bx, x_arr)
            # TODO: we use standard matrix inversion. This is not very efficient. However, we don't
            # yet have a formula for the Sherman-Morrison approximation when a smoothing factor is
            # involved. This is an interesting research direction!
            new_ss_inv = np.linalg.inv(new_ss_arr)
            self._m_arr = new_ss_inv @ (s * ss_arr @ self._m_arr + (1 - s) * bx * y)
            self._ss_arr = new_ss_arr
            self._ss_inv_arr = np.asfortranarray(new_ss_inv)

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
            m_arr = self._m_arr
            for f, v in x.items():
                i = self._idx.get(f)
                if i is not None:
                    y_pred_mean += m_arr[i] * v

        if not with_dist:
            return float(y_pred_mean)

        x_arr = np.fromiter(x.values(), dtype=np.float64, count=len(x))
        ss_inv_arr = self._gather_ss_inv(x.keys())
        # Bishop equation 3.59
        y_pred_var = 1 / self.beta + x_arr @ ss_inv_arr @ x_arr.T

        return proba.Gaussian._from_state(n=1, m=y_pred_mean, sig=y_pred_var**0.5, ddof=0)

    def predict_many(self, X):
        pd = utils.pandas.import_pandas()
        m = np.zeros(len(X.columns), dtype=np.float64)
        for k, f in enumerate(X.columns):
            i = self._idx.get(f)
            if i is not None:
                m[k] = self._m_arr[i]
        return pd.Series(X.values @ m, index=X.index)
