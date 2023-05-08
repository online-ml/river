from __future__ import annotations

from river import stats
from river.stats import _rust_stats


class Quantile(stats.base.Univariate):
    """Running quantile.

    Uses the P² algorithm, which is also known as the "Piecewise-Parabolic quantile estimator".
    The code is inspired by LiveStat's implementation [^2].

    Parameters
    ----------
    q
        Determines which quantile to compute, must be comprised between 0 and 1.

    Examples
    --------

    >>> from river import stats
    >>> import numpy as np

    >>> np.random.seed(42 * 1337)
    >>> mu, sigma = 0, 1
    >>> s = np.random.normal(mu, sigma, 500)

    >>> median = stats.Quantile(0.5)
    >>> for x in s:
    ...    _ = median.update(x)
    >>> print(f'The estimated value of the 50th (median) quantile is {median.get():.4f}')
    The estimated value of the 50th (median) quantile is -0.0275

    >>> print(f'The real value of the 50th (median) quantile is {np.median(s):.4f}')
    The real value of the 50th (median) quantile is -0.0135

    >>> percentile_17 = stats.Quantile(0.17)
    >>> for x in s:
    ...    _ = percentile_17.update(x)
    >>> print(f'The estimated value of the 17th quantile is {percentile_17.get():.4f}')
    The estimated value of the 17th quantile is -0.8652

    >>> print(f'The real value of the 17th quantile is {np.percentile(s,17):.4f}')
    The real value of the 17th quantile is -0.9072

    References
    ----------
    [^1]: [The P² Algorithm for Dynamic Univariateal Computing Calculation of Quantiles and Editor Histograms Without Storing Observations](https://www.cse.wustl.edu/~jain/papers/ftp/psqr.pdf)
    [^2]: [LiveStats](https://github.com/cxxr/LiveStats)
    [^3]: [P² quantile estimator: estimating the median without storing values](https://aakinshin.net/posts/p2-quantile-estimator/)

    """

    # Note for devs, if you want look the pure python implementation here:
    # https://github.com/online-ml/river/blob/40c3190c9d05671ae4c2dc8b76c163ea53a45fb0/river/stats/quantile.py
    def __init__(self, q: float = 0.5):
        super().__init__()
        if not 0 < q < 1:
            raise ValueError("q is not comprised between 0 and 1")
        self._quantile = _rust_stats.RsQuantile(q)
        self._is_updated = False
        self.q = q  # Used by anomaly.QuantileFilter

    def update(self, x):
        self._quantile.update(x)
        if not self._is_updated:
            self._is_updated = True
        return self

    def get(self):
        # HACK: Avoid this following error in `QuantileFilter`
        # panicked at 'index out of bounds: the len is 0 but the index is 0'
        if not self._is_updated:
            return None
        return self._quantile.get()

    def __repr__(self):
        # We surcharge this method to avoid this error on rust side:
        # pyo3_runtime.PanicException: index out of bounds: the len is 0 but the index is 0
        # This error is caused by the `get()` use before the update in the super method.
        value = None
        if self._is_updated:
            value = self.get()
        fmt_value = None if value is None else f"{value:{self._fmt}}".rstrip("0")
        return f"{self.__class__.__name__}: {fmt_value}"


class RollingQuantile(stats.base.RollingUnivariate):
    """Running quantile over a window.

    Parameters
    ----------
    q
        Determines which quantile to compute, must be comprised between 0 and 1.
    window_size
        Size of the window.

    Examples
    --------

    >>> from river import stats

    >>> rolling_quantile = stats.RollingQuantile(
    ...     q=.5,
    ...     window_size=101,
    ... )

    >>> for i in range(1001):
    ...     rolling_quantile = rolling_quantile.update(i)
    ...     if i % 100 == 0:
    ...         print(rolling_quantile.get())
    0.0
    50.0
    150.0
    250.0
    350.0
    450.0
    550.0
    650.0
    750.0
    850.0
    950.0

    References
    ----------
    [^1]: [Left sorted](https://stackoverflow.com/questions/8024571/insert-an-item-into-sorted-list-in-python)

    """

    # Note for devs, if you want look the pure python implementation here:
    # https://github.com/online-ml/river/blob/40c3190c9d05671ae4c2dc8b76c163ea53a45fb0/river/stats/quantile.py
    def __init__(self, q: float, window_size: int):
        super().__init__()
        if not 0 <= q <= 1:
            raise ValueError("q is not comprised between 0 and 1")
        self._rolling_quantile = _rust_stats.RsRollingQuantile(q, window_size)
        self.window_size_value = window_size
        self._is_updated = False

    def update(self, x):
        self._rolling_quantile.update(x)
        if not self._is_updated:
            self._is_updated = True
        return self

    def get(self):
        if not self._is_updated:
            return None
        return self._rolling_quantile.get()

    @property
    def window_size(self):
        return self.window_size_value

    def __repr__(self):
        # We surcharge this method to avoid this error on rust side:
        # pyo3_runtime.PanicException: attempt to subtract with overflow
        # This error is caused by the `get()` use before the update in the super method.
        value = None
        if self._is_updated:
            value = self.get()
        fmt_value = None if value is None else f"{value:{self._fmt}}".rstrip("0")
        return f"{self.__class__.__name__}: {fmt_value}"
