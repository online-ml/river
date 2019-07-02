from . import base


class EWMean(base.Univariate):
    """Exponentially weighted mean.

    Parameters:
        alpha (float): The closer ``alpha`` is to 1 the more the statistic will adapt to recent
            values.

    Attributes:
        mean (float) : The running exponentially weighted mean.

    Example:

        ::

            >>> from creme import stats

            >>> X = [1, 3, 5, 4, 6, 8, 7, 9, 11]
            >>> ewm = stats.EWMean(alpha=0.5)
            >>> for x in X:
            ...     print(ewm.update(x).get())
            1
            2.0
            3.5
            3.75
            4.875
            6.4375
            6.71875
            7.859375
            9.4296875

    References:
        1. `Incremental calculation of weighted mean and variance <http://people.ds.cam.ac.uk/fanf2/hermes/doc/antiforgery/stats.pdf>`_
        2. `Exponential Moving Average on Streaming Data <https://dev.to/nestedsoftware/exponential-moving-average-on-streaming-data-4hhl>`_

    """

    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.mean = 0

    @property
    def name(self):
        return f'ewm_{self.alpha}'

    def update(self, x):
        self.mean = self.alpha * x + (1. - self.alpha) * self.mean if self.mean else x
        return self

    def get(self):
        return self.mean
