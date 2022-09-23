from river import stats
from river.stats import _rust_stats


class EWVar(stats.base.Univariate):
    """Exponentially weighted variance.

    To calculate the variance we use the fact that Var(X) = Mean(x^2) - Mean(x)^2 and internally
    we use the exponentially weighted mean of x/x^2 to calculate this.

    Parameters
    ----------
    alpha
        The closer `alpha` is to 1 the more the statistic will adapt to recent values.

    Attributes
    ----------
    variance : float
        The running exponentially weighted variance.

    Examples
    --------

    >>> from river import stats

    >>> X = [1, 3, 5, 4, 6, 8, 7, 9, 11]
    >>> ewv = stats.EWVar(alpha=0.5)
    >>> for x in X:
    ...     print(ewv.update(x).get())
    0.0
    1.0
    2.75
    1.4375
    1.984375
    3.43359375
    1.7958984375
    2.198974609375
    3.56536865234375

    References
    ----------
    [^1]: [Finch, T., 2009. Incremental calculation of weighted mean and variance. University of Cambridge, 4(11-5), pp.41-42.](https://fanf2.user.srcf.net/hermes/doc/antiforgery/stats.pdf)
    [^2]: [Exponential Moving Average on Streaming Data](https://dev.to/nestedsoftware/exponential-moving-average-on-streaming-data-4hhl)

    """

    # Note for devs, if you want look the pure python implementation here:
    # https://github.com/online-ml/river/blob/40c3190c9d05671ae4c2dc8b76c163ea53a45fb0/river/stats/ewvar.py
    def __init__(self, alpha=0.5):
        if not 0 <= alpha <= 1:
            raise ValueError("q is not comprised between 0 and 1")

        self.alpha = alpha
        self._ewvar = _rust_stats.RsEWVar(alpha)

    @property
    def name(self):
        return f"ewv_{self.alpha}"

    def update(self, x):
        self._ewvar.update(x)
        return self

    def get(self):
        return self._ewvar.get()
