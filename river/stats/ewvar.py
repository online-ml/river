from . import base, ewmean


class EWVar(base.Univariate):
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
    0
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

    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.mean = ewmean.EWMean(alpha=alpha)
        self.sq_mean = ewmean.EWMean(alpha=alpha)

    @property
    def name(self):
        return f"ewv_{self.alpha}"

    def update(self, x):
        self.mean.update(x)
        self.sq_mean.update(x ** 2)
        return self

    def get(self):
        return self.sq_mean.get() - self.mean.get() ** 2
