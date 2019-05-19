from . import base
from .ewmean import EWMean


class EWVar(base.Univariate):
    """Exponentially weighted variance.

    Parameters:
        alpha (float): The closer ``alpha`` is to 1 the more the statistic will adapt to recent
            values.

    Attributes:
        variance (float) : The running exponentially weighted variance.

    Example:

        ::

            >>> from creme import stats

            >>> X = [1, 3, 5, 4, 6, 8, 7, 9, 11]
            >>> ewv = EWVar(alpha=0.5)
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

    References:

        1. `Incremental calculation of weighted mean and variance <http://people.ds.cam.ac.uk/fanf2/hermes/doc/antiforgery/stats.pdf>`_
        2. `Exponential Moving Average on Streaming Data <https://dev.to/nestedsoftware/exponential-moving-average-on-streaming-data-4hhl>`_
        3. `Pandas User Guide <http://pandas.pydata.org/pandas-docs/stable/user_guide/computation.html>`

    """

    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.n = 0
        self.mean = EWMean(alpha=alpha)
        self.sq_mean = EWMean(alpha=alpha)

    @property
    def name(self):
        return f'ewv_{self.alpha}'

    def update(self, x):
        self.mean.update(x)
        self.sq_mean.update(x**2)
        self.n += 1
        return self

    def get(self):
        return self.sq_mean.get() - self.mean.get()**2
