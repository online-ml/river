from river import stats
from river.stats import _rust_stats


class EWMean(stats.base.Univariate):
    """Exponentially weighted mean.

    Parameters
    ----------
    alpha
        The closer `alpha` is to 1 the more the statistic will adapt to recent values.

    Attributes
    ----------
    mean : float
        The running exponentially weighted mean.

    Examples
    --------

    >>> from river import stats

    >>> X = [1, 3, 5, 4, 6, 8, 7, 9, 11]
    >>> ewm = stats.EWMean(alpha=0.5)
    >>> for x in X:
    ...     print(ewm.update(x).get())
    1.0
    2.0
    3.5
    3.75
    4.875
    6.4375
    6.71875
    7.859375
    9.4296875

    References
    ----------
    [^1]: [Finch, T., 2009. Incremental calculation of weighted mean and variance. University of Cambridge, 4(11-5), pp.41-42.](https://fanf2.user.srcf.net/hermes/doc/antiforgery/stats.pdf)
    [^2]: [Exponential Moving Average on Streaming Data](https://dev.to/nestedsoftware/exponential-moving-average-on-streaming-data-4hhl)

    """

    # Note for devs, if you want look the pure python implementation here:
    # https://github.com/online-ml/river/blob/40c3190c9d05671ae4c2dc8b76c163ea53a45fb0/river/stats/ewmean.py
    def __init__(self, alpha=0.5):
        if not 0 <= alpha <= 1:
            raise ValueError("q is not comprised between 0 and 1")
        self.alpha = alpha
        self._ewmean = _rust_stats.RsEWMean(alpha)
        self.mean = 0

    @property
    def name(self):
        return f"ewm_{self.alpha}"

    def update(self, x):
        self._ewmean.update(x)
        return self

    def get(self):
        return self._ewmean.get()
