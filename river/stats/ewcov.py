from __future__ import annotations

from river import stats


class EWCov(stats.base.Bivariate):
    """Exponentially weighted covariance.

    This is the bivariate counterpart of `stats.EWVar`. It tracks the covariance between two
    variables while giving more weight to recent observations, which is what you want when the
    relationship between the variables drifts over time (e.g. the co-movement of two asset
    returns during a changing market regime).

    Internally it uses the identity $Cov(x, y) = E[xy] - E[x]E[y]$, with each expectation
    estimated by an exponentially weighted mean (`stats.EWMean`). Using the same `fading_factor`
    on the diagonal recovers `stats.EWVar` exactly.

    Parameters
    ----------
    fading_factor
        The closer `fading_factor` is to 1 the more the statistic will adapt to recent values.

    Examples
    --------

    >>> from river import stats

    >>> x = [1, 3, 5, 4]
    >>> y = [2, 4, 3, 6]

    >>> ewcov = stats.EWCov(fading_factor=0.5)
    >>> for xi, yi in zip(x, y):
    ...     ewcov.update(xi, yi)
    ...     print(ewcov.get())
    0.0
    1.0
    0.5
    0.625

    References
    ----------
    [^1]: [Finch, T., 2009. Incremental calculation of weighted mean and variance. University of Cambridge, 4(11-5), pp.41-42.](https://fanf2.user.srcf.net/hermes/doc/antiforgery/stats.pdf)
    [^2]: [Exponential Moving Average on Streaming Data](https://dev.to/nestedsoftware/exponential-moving-average-on-streaming-data-4hhl)

    """

    def __init__(self, fading_factor: float = 0.5) -> None:
        if not 0 <= fading_factor <= 1:
            raise ValueError("fading_factor is not comprised between 0 and 1")
        self.fading_factor: float = fading_factor
        self._mean_x: stats.EWMean = stats.EWMean(fading_factor)
        self._mean_y: stats.EWMean = stats.EWMean(fading_factor)
        self._mean_xy: stats.EWMean = stats.EWMean(fading_factor)

    @property
    def name(self) -> str:
        return f"ewcov_{self.fading_factor}"

    def update(self, x: float, y: float) -> None:
        self._mean_x.update(x)
        self._mean_y.update(y)
        self._mean_xy.update(x * y)

    def get(self) -> float:
        return self._mean_xy.get() - self._mean_x.get() * self._mean_y.get()
