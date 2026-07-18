from __future__ import annotations

import typing

import numpy as np

from river.utils import dataframe as dataframe_utils

from .emp import SymmetricMatrix

if typing.TYPE_CHECKING:
    from narwhals.stable.v2.typing import IntoDataFrame

__all__ = [
    "EwaCovariance",
    "EwaPrecision",
    "LedoitWolfCovariance",
    "OASCovariance",
    "ShrunkCovariance",
]

_EPS = 1e-12


class _EWMatrix(SymmetricMatrix):
    """Array-backed exponentially weighted matrix estimator with a dict-native interface.

    Shared engine for the exponentially weighted covariance and precision estimators. It keeps a
    running exponentially weighted mean vector (using the same convention as `stats.EWMean`) plus a
    matrix-valued state, stored as dense numpy arrays with a feature to index map (the same pattern
    as `covariance.EmpiricalPrecision`). A consistent set of features is assumed across
    observations, mirroring the usual covariance-estimation setting; new features seen mid-stream
    grow the matrix.
    """

    def __init__(self, fading_factor: float = 0.5):
        if not 0 <= fading_factor <= 1:
            raise ValueError("fading_factor is not comprised between 0 and 1")
        self.fading_factor = float(fading_factor)
        self._features: list = []
        self._idx: dict = {}
        self._mean = np.zeros(0, dtype=np.float64)
        self._initialized = np.zeros(0, dtype=bool)
        self._n = 0
        self._matrix_cache: np.ndarray | None = None

    # --------------------------------------------------------------- internals
    def _on_grow(self, old_dim: int, new_dim: int) -> None:
        """Resize the subclass's matrix state when new features appear."""
        raise NotImplementedError

    def _grow(self, new_keys: list) -> None:
        old_dim = len(self._features)
        for k in new_keys:
            self._idx[k] = len(self._features)
            self._features.append(k)
        new_dim = len(self._features)
        mean = np.zeros(new_dim, dtype=np.float64)
        mean[:old_dim] = self._mean
        init = np.zeros(new_dim, dtype=bool)
        init[:old_dim] = self._initialized
        self._mean, self._initialized = mean, init
        self._on_grow(old_dim, new_dim)

    def _vector(self, x: dict) -> np.ndarray:
        new_keys = [k for k in x if k not in self._idx]
        if new_keys:
            self._grow(new_keys)
        try:
            return np.array([x[f] for f in self._features], dtype=np.float64)
        except KeyError as e:
            raise ValueError(
                f"observation is missing feature {e.args[0]!r}; the exponentially weighted "
                "estimators assume a consistent set of features"
            ) from e

    def _fresh(self) -> np.ndarray:
        return ~self._initialized

    def _blend_mean(self, v: np.ndarray, fresh: np.ndarray) -> np.ndarray:
        f = self.fading_factor
        # Seed freshly-seen features with the observed value (like stats.EWMean's first step).
        if fresh.any():
            return np.where(fresh, v, (1 - f) * self._mean + f * v)
        return (1 - f) * self._mean + f * v

    def _learn_vector(self, v: np.ndarray) -> None:
        raise NotImplementedError

    def _to_matrix_array(self) -> np.ndarray:
        raise NotImplementedError

    def _matrix_array(self) -> np.ndarray:
        if self._matrix_cache is None:
            self._matrix_cache = self._to_matrix_array()
        return self._matrix_cache

    # -------------------------------------------------------------- public API
    def update(self, x: dict):
        """Update with a single sample.

        Parameters
        ----------
        x
            A sample.

        """
        self._learn_vector(self._vector(x))

    def update_many(self, X: IntoDataFrame):
        """Update with a dataframe of samples.

        Any [narwhals](https://github.com/narwhals-dev/narwhals)-compatible eager dataframe
        (pandas, polars, pyarrow, ...) is accepted. The result is identical to feeding the rows
        one at a time with `update`, in row order.

        Parameters
        ----------
        X
            A dataframe of samples.

        """
        frame = dataframe_utils.into_frame(X)
        columns = list(frame.columns)
        new_keys = [k for k in columns if k not in self._idx]
        if new_keys:
            self._grow(new_keys)
        ids = [self._idx[c] for c in columns]
        # Lay each row out in the estimator's feature order before the sequential update.
        arr = dataframe_utils.to_numpy(frame)
        layout = np.empty((arr.shape[0], len(self._features)), dtype=np.float64)
        layout[:] = self._mean  # carry forward known features absent from this batch
        layout[:, ids] = arr
        for row in layout:
            self._learn_vector(row)

    @property
    def matrix(self) -> dict:
        if not self._features:
            return {}
        arr = self._matrix_array()
        out = {}
        for ai, fa in enumerate(self._features):
            for bi in range(ai, len(self._features)):
                fb = self._features[bi]
                out[min((fa, fb), (fb, fa))] = float(arr[ai, bi])
        return out

    def __getitem__(self, key):
        i, j = key
        ai = self._idx.get(i)
        bi = self._idx.get(j)
        if ai is None or bi is None:
            raise KeyError(key)
        return float(self._matrix_array()[ai, bi])


class _EWCovariance(_EWMatrix):
    """Exponentially weighted covariance engine.

    Tracks an exponentially weighted mean vector and second-moment matrix and exposes the
    covariance ``Σ = E[xxᵀ] - E[x]E[x]ᵀ``. The diagonal matches `stats.EWVar` and the off-diagonals
    match `stats.EWCov`. Shrinkage subclasses override `_to_matrix_array`.
    """

    def __init__(self, fading_factor: float = 0.5):
        super().__init__(fading_factor)
        self._M2 = np.zeros((0, 0), dtype=np.float64)

    def _on_grow(self, old_dim: int, new_dim: int) -> None:
        M2 = np.zeros((new_dim, new_dim), dtype=np.float64)
        M2[:old_dim, :old_dim] = self._M2
        self._M2 = M2

    def _before_update(self, v: np.ndarray) -> None:
        """Hook called with the new observation before the state is updated (mean/M2 are old)."""

    def _learn_vector(self, v: np.ndarray) -> None:
        self._before_update(v)
        f = self.fading_factor
        fresh = self._fresh()
        outer = np.outer(v, v)
        self._mean = self._blend_mean(v, fresh)
        if fresh.any():
            fresh_pair = fresh[:, None] | fresh[None, :]
            self._M2 = np.where(fresh_pair, outer, (1 - f) * self._M2 + f * outer)
        else:
            self._M2 = (1 - f) * self._M2 + f * outer
        self._initialized[fresh] = True
        self._n += 1
        self._matrix_cache = None

    def _raw_cov(self) -> np.ndarray:
        return self._M2 - np.outer(self._mean, self._mean)

    def _to_matrix_array(self) -> np.ndarray:
        return self._raw_cov()


class EwaCovariance(_EWCovariance):
    """Exponentially weighted covariance matrix.

    A recency-weighted estimate of the covariance: each new observation is blended into the
    estimate with weight ``fading_factor``, so the influence of past observations decays
    geometrically. This is the streaming analogue of the RiskMetrics covariance and the matrix
    counterpart of `stats.EWVar` / `stats.EWCov` (the diagonal is exactly `stats.EWVar` and each
    off-diagonal is exactly `stats.EWCov`).

    **When to use it.** Reach for this over `EmpiricalCovariance` when the relationships between
    your variables change over time. The textbook example is asset returns, whose volatilities and
    correlations move with the market regime: a plain empirical covariance weights a return from
    years ago the same as yesterday's, whereas an exponentially weighted one forgets the distant
    past so the risk estimate tracks current conditions. Larger `fading_factor` reacts faster
    (shorter memory); smaller is smoother.

    Parameters
    ----------
    fading_factor
        The closer `fading_factor` is to 1 the more weight recent observations carry. The
        effective memory is roughly ``1 / fading_factor`` observations.

    Examples
    --------

    We estimate the covariance of daily returns (in percent) for a few stocks from the
    `datasets.SP500Stocks` dataset.

    >>> from river import covariance, datasets

    >>> tickers = ["AAPL", "JPM", "XOM"]
    >>> cov = covariance.EwaCovariance(fading_factor=0.02)
    >>> for x, _ in datasets.SP500Stocks():
    ...     cov.update({t: x[t] for t in tickers})
    >>> cov
           AAPL    JPM     XOM
    AAPL   1.944   0.766   0.760
     JPM   0.766   1.492   0.934
     XOM   0.760   0.934   1.705

    There is also an `update_many` method to process mini-batches. It accepts any
    narwhals-compatible dataframe and yields the same result as feeding the rows one by one.

    >>> import pandas as pd
    >>> returns = pd.DataFrame(x for x, _ in datasets.SP500Stocks())[tickers]
    >>> cov = covariance.EwaCovariance(fading_factor=0.02)
    >>> cov.update_many(returns)
    >>> cov
           AAPL    JPM     XOM
    AAPL   1.944   0.766   0.760
     JPM   0.766   1.492   0.934
     XOM   0.760   0.934   1.705

    Individual entries are accessible by key:

    >>> cov["AAPL", "JPM"]
    0.766...

    References
    ----------
    [^1]: [RiskMetrics Technical Document (J.P. Morgan, 1996)](https://www.msci.com/documents/10199/5915b101-4206-4ba0-aee2-3449d5c7e95a)

    """


class LedoitWolfCovariance(_EWCovariance):
    """Online Ledoit-Wolf shrinkage covariance.

    Shrinks the exponentially weighted sample covariance towards a scaled identity (a sphere) by a
    data-driven intensity, following Ledoit & Wolf (2004). The shrinkage intensity is estimated
    online from the dispersion of the per-observation scatter around the running covariance, and is
    recomputed on read; the per-step update stays O(d^2) with no stored window.

    **When to use it.** The raw sample covariance is a poor estimate when the number of variables
    is large relative to the (effective) number of observations: it is noisy and often
    ill-conditioned or singular, which wrecks anything that inverts it (portfolio optimisation,
    Mahalanobis distances, Gaussian likelihoods). Shrinkage trades a little bias for a large
    reduction in variance, producing a well-conditioned, invertible matrix. Ledoit-Wolf picks the
    shrinkage intensity for you, so it is a strong default when there are many assets relative to
    clean history.

    Parameters
    ----------
    fading_factor
        Recency weight of the underlying exponentially weighted covariance. The effective sample
        size is roughly ``1 / fading_factor``.

    Examples
    --------

    On the ten stocks of `datasets.SP500Stocks` with a short effective memory, the raw covariance is
    poorly conditioned; Ledoit-Wolf shrinkage tames it.

    >>> import numpy as np
    >>> from river import covariance, datasets

    >>> ewa = covariance.EwaCovariance(fading_factor=0.05)
    >>> lw = covariance.LedoitWolfCovariance(fading_factor=0.05)
    >>> for x, _ in datasets.SP500Stocks():
    ...     ewa.update(x)
    ...     lw.update(x)

    >>> def condition_number(cov):
    ...     names = sorted({i for i, _ in cov.matrix})
    ...     M = np.array([[cov[i, j] for j in names] for i in names])
    ...     return np.linalg.cond(M)

    >>> condition_number(ewa) > condition_number(lw)
    np.True_

    References
    ----------
    [^1]: [Ledoit, O. and Wolf, M., 2004. A well-conditioned estimator for large-dimensional covariance matrices. Journal of Multivariate Analysis, 88(2), pp.365-411.](https://www.ledoit.net/honey.pdf)

    """

    def __init__(self, fading_factor: float = 0.5):
        super().__init__(fading_factor)
        self._pi_bar = 0.0  # running dispersion of the per-sample scatter about the covariance

    def _before_update(self, v: np.ndarray) -> None:
        d = len(v)
        if d == 0:
            return
        dev = v - self._mean
        scatter = np.outer(dev, dev)
        cov_old = self._raw_cov()
        q = float(((scatter - cov_old) ** 2).sum() / d)
        f = self.fading_factor
        self._pi_bar = (1 - f) * self._pi_bar + f * q

    def _to_matrix_array(self) -> np.ndarray:
        S = self._raw_cov()
        d = S.shape[0]
        mu = np.trace(S) / d
        identity = np.eye(d)
        disp = float(((S - mu * identity) ** 2).sum() / d)
        if disp <= _EPS or self._pi_bar <= 0:
            return S
        # Effective sample size of the exponentially weighted estimator is ~ 1 / fading_factor.
        b2 = min(self._pi_bar * self.fading_factor, disp)
        intensity = float(np.clip(b2 / disp, 0.0, 1.0))
        return (1 - intensity) * S + intensity * mu * identity


class OASCovariance(_EWCovariance):
    """Online Oracle Approximating Shrinkage (OAS) covariance.

    Like `LedoitWolfCovariance`, OAS shrinks the exponentially weighted sample covariance towards a
    scaled identity, but uses the Chen, Wiesel, Eldar & Hero (2010) shrinkage intensity, which is
    often better conditioned for approximately Gaussian data. The intensity is a closed form in the
    traces of the running covariance, applied on read; the per-step update is a plain O(d^2)
    exponentially weighted covariance update.

    **When to use it.** The same high-dimensional / few-sample situations as
    `LedoitWolfCovariance`. OAS tends to shrink slightly more aggressively and is a good choice when
    the data are close to Gaussian; otherwise the two are interchangeable and worth comparing.

    Parameters
    ----------
    fading_factor
        Recency weight of the underlying exponentially weighted covariance. The effective sample
        size is roughly ``1 / fading_factor``.

    Examples
    --------

    >>> import numpy as np
    >>> from river import covariance, datasets

    >>> oas = covariance.OASCovariance(fading_factor=0.05)
    >>> for x, _ in datasets.SP500Stocks():
    ...     oas.update(x)

    The shrinkage keeps the matrix positive-definite and invertible:

    >>> names = sorted({i for i, _ in oas.matrix})
    >>> M = np.array([[oas[i, j] for j in names] for i in names])
    >>> bool(np.all(np.linalg.eigvalsh(M) > 0))
    True

    References
    ----------
    [^1]: [Chen, Y., Wiesel, A., Eldar, Y.C. and Hero, A.O., 2010. Shrinkage algorithms for MMSE covariance estimation. IEEE Transactions on Signal Processing, 58(10), pp.5016-5029.](https://arxiv.org/abs/0907.4698)

    """

    def _to_matrix_array(self) -> np.ndarray:
        S = self._raw_cov()
        d = S.shape[0]
        n = max(1.0 / self.fading_factor, 2.0)
        tr = np.trace(S)
        tr2 = np.trace(S @ S)
        mu = tr / d
        num = (1.0 - 2.0 / d) * tr2 + tr * tr
        den = (n + 1.0 - 2.0 / d) * (tr2 - tr * tr / d)
        rho = 1.0 if den <= 0 else min(max(num / den, 0.0), 1.0)
        return (1.0 - rho) * S + rho * mu * np.eye(d)


class ShrunkCovariance(_EWCovariance):
    """Fixed-intensity shrinkage covariance, towards a constant-correlation or identity target.

    Where `LedoitWolfCovariance` and `OASCovariance` estimate the shrinkage intensity from the
    data, `ShrunkCovariance` uses a fixed `delta` (transparent and predictable, mirroring
    `sklearn.covariance.ShrunkCovariance`) and offers a choice of target.

    **When to use it.** When you want explicit, reproducible control over how much regularisation is
    applied rather than letting the data decide. The **constant-correlation** target (every pair
    shrunk towards the average sample correlation) is the finance-relevant default: assets share a
    positive baseline correlation, so pulling towards that is more sensible than pulling towards the
    zero-correlation identity target.

    Parameters
    ----------
    fading_factor
        Recency weight of the underlying exponentially weighted covariance.
    delta
        Shrinkage intensity in [0, 1] (0 = the raw covariance, 1 = the target).
    target
        Either ``"constant_correlation"`` (default) or ``"identity"``.

    Examples
    --------

    >>> from river import covariance, datasets

    >>> tickers = ["AAPL", "JPM", "XOM"]
    >>> cov = covariance.ShrunkCovariance(fading_factor=0.02, delta=0.3)
    >>> for x, _ in datasets.SP500Stocks():
    ...     cov.update({t: x[t] for t in tickers})
    >>> cov
           AAPL    JPM     XOM
    AAPL   1.944   0.784   0.797
     JPM   0.784   1.492   0.885
     XOM   0.797   0.885   1.705

    With ``delta=0`` the estimator reduces to a plain exponentially weighted covariance, and with
    ``delta=1`` it returns the target exactly.

    References
    ----------
    [^1]: [Ledoit, O. and Wolf, M., 2003. Improved estimation of the covariance matrix of stock returns with an application to portfolio selection. Journal of Empirical Finance, 10(5), pp.603-621.](https://www.ledoit.net/ole2.pdf)

    """

    def __init__(
        self,
        fading_factor: float = 0.5,
        delta: float = 0.1,
        target: str = "constant_correlation",
    ):
        if target not in ("constant_correlation", "identity"):
            raise ValueError("target must be 'constant_correlation' or 'identity'")
        super().__init__(fading_factor)
        self.delta = delta
        self.target = target

    def _to_matrix_array(self) -> np.ndarray:
        S = self._raw_cov()
        d = S.shape[0]
        if self.target == "identity":
            target = (np.trace(S) / d) * np.eye(d)
        else:  # constant_correlation
            sd = np.sqrt(np.maximum(np.diag(S), _EPS))
            corr = S / np.outer(sd, sd)
            off = corr[~np.eye(d, dtype=bool)]
            rbar = float(np.mean(off)) if off.size else 0.0
            target = rbar * np.outer(sd, sd)
            np.fill_diagonal(target, np.diag(S))  # preserve the variances
        return (1.0 - self.delta) * S + self.delta * target


class EwaPrecision(_EWMatrix):
    """Exponentially weighted precision (inverse covariance) matrix.

    The recency-weighted analogue of `EmpiricalPrecision`. It maintains the inverse of an
    exponentially weighted second-moment matrix online via a forgetting-factor Sherman-Morrison
    update (the recursive-least-squares trick), and applies the mean centering on read. It is
    genuinely online: the per-step cost is O(d^2) and the matrix is never explicitly inverted.

    **When to use it.** Several methods need the *precision* matrix rather than the covariance:
    the Mahalanobis distance (anomaly detection), the Gaussian log-likelihood, and the weights of a
    Gaussian graphical model. Use this when those quantities must track a non-stationary stream,
    where inverting a stale covariance would lag. Like `EmpiricalPrecision`, the result is not
    guaranteed identical to inverting the matching covariance (there is a decaying identity prior),
    but the difference shrinks as observations accumulate. Requires ``0 < fading_factor < 1``.

    Parameters
    ----------
    fading_factor
        Recency weight of the most recent observation. The effective memory is roughly
        ``1 / fading_factor`` observations.

    Examples
    --------

    >>> from river import covariance, datasets

    >>> tickers = ["AAPL", "JPM", "XOM"]
    >>> prec = covariance.EwaPrecision(fading_factor=0.02)
    >>> for x, _ in datasets.SP500Stocks():
    ...     prec.update({t: x[t] for t in tickers})
    >>> prec
           AAPL     JPM      XOM
    AAPL    0.676   -0.241   -0.169
     JPM   -0.241    1.105   -0.498
     XOM   -0.169   -0.498    0.934

    Up to the decaying prior, this approximates the inverse of the matching `EwaCovariance`:

    >>> import numpy as np
    >>> cov = covariance.EwaCovariance(fading_factor=0.02)
    >>> for x, _ in datasets.SP500Stocks():
    ...     cov.update({t: x[t] for t in tickers})
    >>> S = np.array([[cov[i, j] for j in tickers] for i in tickers])
    >>> P = np.array([[prec[i, j] for j in tickers] for i in tickers])
    >>> bool(np.allclose(P @ S, np.eye(3), atol=1e-6))
    True

    References
    ----------
    [^1]: [Online Estimation of the Inverse Covariance Matrix - Markus Thill](https://markusthill.github.io/math/stats/ml/online-estimation-of-the-inverse-covariance-matrix/)
    [^2]: [Recursive least squares filter](https://en.wikipedia.org/wiki/Recursive_least_squares_filter)

    """

    def __init__(self, fading_factor: float = 0.5):
        if not 0 < fading_factor < 1:
            raise ValueError("fading_factor must be strictly between 0 and 1")
        super().__init__(fading_factor)
        self._Pm = np.zeros((0, 0), dtype=np.float64)  # inverse of the EW second-moment matrix

    def _on_grow(self, old_dim: int, new_dim: int) -> None:
        # New features start from an identity prior on the second-moment matrix.
        Pm = np.eye(new_dim, dtype=np.float64)
        Pm[:old_dim, :old_dim] = self._Pm
        self._Pm = Pm

    def _learn_vector(self, v: np.ndarray) -> None:
        f = self.fading_factor
        fresh = self._fresh()
        # Forgetting-factor Sherman-Morrison update of inv(M2), where M2 <- (1-f) M2 + f v vᵀ.
        Pm = self._Pm / (1 - f)
        Pv = Pm @ v
        denom = 1.0 + f * float(v @ Pv)
        Pm = Pm - f * np.outer(Pv, Pv) / denom
        self._Pm = 0.5 * (Pm + Pm.T)
        self._mean = self._blend_mean(v, fresh)
        self._initialized[fresh] = True
        self._n += 1
        self._matrix_cache = None

    def _to_matrix_array(self) -> np.ndarray:
        # precision = inv(M2 - mean meanᵀ), via a rank-one Sherman-Morrison downdate of inv(M2).
        Pm = self._Pm
        m = self._mean
        Pm_m = Pm @ m
        denom = 1.0 - float(m @ Pm_m)
        prec = Pm if denom <= _EPS else Pm + np.outer(Pm_m, Pm_m) / denom
        return 0.5 * (prec + prec.T)
