from . import base
from . import quantile


class IQR(base.Univariate):
    """Computes the interquartile range.

    Parameters:
        q_inf (float): Desired inferior quantile, must be between 0 and 1. Defaults to `0.25`.
        q_sup (float): Desired superior quantile, must be between 0 and 1. Defaults to `0.75`.

    Example:

        ::

            >>> from creme import stats
            >>> iqr = stats.IQR(
            ...     q_inf=0.25,
            ...     q_sup=0.75
            ... )

            >>> for i in range(0, 1001):
            ...     iqr = iqr.update(i)
            ...     if i % 100 == 0:
            ...         print(iqr.get())
            0
            50
            100
            150
            200
            250
            300
            350
            400
            450
            500
    """

    def __init__(self, q_inf=0.25, q_sup=0.75):
        if q_inf >= q_sup:
            raise ValueError('q_inf must be strictly less than q_sup')
        self.quantile_inf = quantile.Quantile(quantile=q_inf)
        self.quantile_sup = quantile.Quantile(quantile=q_sup)

    def update(self, x):
        self.quantile_inf.update(x)
        self.quantile_sup.update(x)
        return self

    def get(self):
        return self.quantile_sup.get() - self.quantile_inf.get()


class RollingIQR(base.RollingUnivariate):
    """Computes the rolling interquartile range.

    Parameters:
        window_size (int): Size of the window.
        q_inf (float): Desired inferior quantile, must be between 0 and 1. Defaults to `0.25`.
        q_sup (float): Desired superior quantile, must be between 0 and 1. Defaults to `0.75`.

    Example:

        ::

            >>> from creme import stats
            >>> rolling_iqr = stats.RollingIQR(
            ...     q_inf=0.25,
            ...     q_sup=0.75,
            ...     window_size=100
            ... )

            >>> for i in range(0, 1001):
            ...     rolling_iqr = iqr.update(i)
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

    def __init__(self, window_size, q_inf=0.25, q_sup=0.75):
        if q_inf >= q_sup:
            raise ValueError('q_inf must be strictly less than q_sup')
        self.quantile_inf = quantile.RollingQuantile(
            window_size=window_size, quantile=q_inf)
        self.quantile_sup = quantile.RollingQuantile(
            window_size=window_size, quantile=q_sup)

    def update(self, x):
        self.quantile_inf.update(x)
        self.quantile_sup.update(x)
        return self

    def get(self):
        return self.quantile_sup.get() - self.quantile_inf.get()
