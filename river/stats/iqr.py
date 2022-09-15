from river import stats
from river.stats import _rust_stats


class IQR(stats.base.Univariate):
    """Computes the interquartile range.

    Parameters
    ----------
    q_inf
        Desired inferior quantile, must be between 0 and 1. Defaults to `0.25`.
    q_sup
        Desired superior quantile, must be between 0 and 1. Defaults to `0.75`.

    Examples
    --------

    >>> from river import stats

    >>> iqr = stats.IQR(q_inf=0.25, q_sup=0.75)

    >>> for i in range(0, 1001):
    ...     iqr = iqr.update(i)
    ...     if i % 100 == 0:
    ...         print(iqr.get())
    0.0
    50.0
    100.0
    150.0
    200.0
    250.0
    300.0
    350.0
    400.0
    450.0
    500.0

    """

    # Note for devs, if you want look the pure python implementation here:
    # https://github.com/online-ml/river/blob/40c3190c9d05671ae4c2dc8b76c163ea53a45fb0/river/stats/iqr.py

    def __init__(self, q_inf=0.25, q_sup=0.75):
        super().__init__()
        if q_inf >= q_sup:
            raise ValueError("q_inf must be strictly less than q_sup")
        self.q_inf = q_inf
        self.q_sup = q_sup
        self._iqr = _rust_stats.RsIQR(self.q_inf, self.q_sup)
        self._is_updated = False

    @property
    def name(self):
        return f"{self.__class__.__name__}_{self.q_inf}_{self.q_sup}"

    def update(self, x):
        self._iqr.update(x)
        if not self._is_updated:
            self._is_updated = True
        return self

    def get(self):
        return self._iqr.get()

    def __repr__(self):
        # We surcharge this method to avoid this error on rust side:
        # pyo3_runtime.PanicException: index out of bounds: the len is 0 but the index is 0
        # This error is caused by the `get()` use before the update in the super method.
        value = None
        if self._is_updated:
            value = self.get()
        fmt_value = None if value is None else f"{value:{self._fmt}}".rstrip("0")
        return f"{self.__class__.__name__}: {fmt_value}"


class RollingIQR(stats.base.RollingUnivariate):
    """Computes the rolling interquartile range.

    Parameters
    ----------
    window_size
        Size of the window.
    q_inf
        Desired inferior quantile, must be between 0 and 1. Defaults to `0.25`.
    q_sup
        Desired superior quantile, must be between 0 and 1. Defaults to `0.75`.

    Examples
    --------

    >>> from river import stats
    >>> rolling_iqr = stats.RollingIQR(
    ...     q_inf=0.25,
    ...     q_sup=0.75,
    ...     window_size=101
    ... )

    >>> for i in range(0, 1001):
    ...     rolling_iqr = rolling_iqr.update(i)
    ...     if i % 100 == 0:
    ...         print(rolling_iqr.get())
    0.0
    50.0
    50.0
    50.0
    50.0
    50.0
    50.0
    50.0
    50.0
    50.0
    50.0

    """

    # Note for devs, if you want look the pure python implementation here:
    # https://github.com/online-ml/river/blob/40c3190c9d05671ae4c2dc8b76c163ea53a45fb0/river/stats/iqr.py
    def __init__(self, window_size: int, q_inf=0.25, q_sup=0.75):
        if q_inf >= q_sup:
            raise ValueError("q_inf must be strictly less than q_sup")
        self.q_inf = q_inf
        self.q_sup = q_sup
        self._rolling_iqr = _rust_stats.RsRollingIQR(q_inf, q_sup, window_size)
        self.window_size_value = window_size
        self._is_updated = False

    def update(self, x):
        self._rolling_iqr.update(x)
        if not self._is_updated:
            self._is_updated = True
        return self

    def get(self):
        # HACK: Avoid crash if get is called before update
        # panicked at 'index out of bounds: the len is 0 but the index is 0'
        if not self._is_updated:
            return None
        return self._rolling_iqr.get()

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
