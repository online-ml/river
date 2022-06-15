from river import stats


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
        self.quantile_inf = stats.Quantile(q=self.q_inf)
        self.quantile_sup = stats.Quantile(q=self.q_sup)

    @property
    def name(self):
        return f"{self.__class__.__name__}_{self.q_inf}_{self.q_sup}"

    def update(self, x):
        self.quantile_inf.update(x)
        self.quantile_sup.update(x)
        return self

    def get(self):
        q_sup = self.quantile_sup.get()
        if q_sup is None:
            return None
        q_inf = self.quantile_inf.get()
        if q_inf is None:
            return None
        return q_sup - q_inf


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

    def __init__(self, window_size: int, q_inf=0.25, q_sup=0.75):
        if q_inf >= q_sup:
            raise ValueError("q_inf must be strictly less than q_sup")
        self.q_inf = q_inf
        self.q_sup = q_sup
        self.quantile_inf = stats.RollingQuantile(q=self.q_inf, window_size=window_size)
        self.quantile_sup = stats.RollingQuantile(q=self.q_sup, window_size=window_size)

    @property
    def window_size(self):
        return self.quantile_inf.window_size

    @property
    def name(self):
        return f"rolling_{self.__class__.__name__}_{self.q_inf}_{self.q_sup}"

    def update(self, x):
        self.quantile_inf.update(x)
        self.quantile_sup.update(x)
        return self

    def get(self):
        q_sup = self.quantile_sup.get()
        if q_sup is None:
            return None
        q_inf = self.quantile_inf.get()
        if q_inf is None:
            return None
        return q_sup - q_inf
