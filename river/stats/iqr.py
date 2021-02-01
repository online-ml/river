from .. import utils
from . import base, quantile


class IQR(base.Univariate):
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
    0
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

    def __init__(self, q_inf=0.25, q_sup=0.75):
        if q_inf >= q_sup:
            raise ValueError("q_inf must be strictly less than q_sup")
        self.q_inf = q_inf
        self.q_sup = q_sup
        self.quantile_inf = quantile.Quantile(q=self.q_inf)
        self.quantile_sup = quantile.Quantile(q=self.q_sup)

    @property
    def name(self):
        return f"{self.__class__.__name__}_{self.q_inf}_{self.q_sup}"

    def update(self, x):
        self.quantile_inf.update(x)
        self.quantile_sup.update(x)
        return self

    def get(self):
        return self.quantile_sup.get() - self.quantile_inf.get()


class RollingIQR(base.RollingUnivariate, utils.SortedWindow):
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
    ...     window_size=100
    ... )

    >>> for i in range(0, 1001):
    ...     rolling_iqr = rolling_iqr.update(i)
    ...     if i % 100 == 0:
    ...         print(rolling_iqr.get())
    0
    50
    50
    50
    50
    50
    50
    50
    50
    50
    50

    """

    def __init__(self, window_size: int, q_inf=0.25, q_sup=0.75):
        super().__init__(size=window_size)
        if q_inf >= q_sup:
            raise ValueError("q_inf must be strictly less than q_sup")
        self.q_inf = q_inf
        self.q_sup = q_sup
        self.quantile_inf = quantile.RollingQuantile(
            q=self.q_inf, window_size=window_size
        )
        self.quantile_sup = quantile.RollingQuantile(
            q=self.q_sup, window_size=window_size
        )

    @property
    def name(self):
        return f"rolling_{self.__class__.__name__}_{self.q_inf}_{self.q_sup}"

    def update(self, x):
        self.quantile_inf.update(x)
        self.quantile_sup.update(x)
        return self

    def get(self):
        return self.quantile_sup.get() - self.quantile_inf.get()
