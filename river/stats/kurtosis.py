from . import moments


class Kurtosis(moments.CentralMoments):
    """Running kurtosis using Welford's algorithm.

    Parameters
    ----------
    bias
        If `False`, then the calculations are corrected for statistical bias.

    Examples
    --------

    >>> import river.stats
    >>> import scipy.stats
    >>> import numpy as np

    >>> np.random.seed(42)
    >>> X = np.random.normal(loc=0, scale=1, size=10)

    >>> kurtosis = river.stats.Kurtosis(bias=False)
    >>> for x in X:
    ...     print(kurtosis.update(x).get())
    -3
    -2.0
    -1.5
    1.4130027920707047
    0.15367976585756438
    0.46142633246812653
    -1.620647789230658
    -1.3540178492487054
    -1.2310268787102745
    -0.9490372374384453

    >>> for i in range(1, len(X)+1):
    ...     print(scipy.stats.kurtosis(X[:i], bias=False))
    -3.0
    -2.0
    -1.4999999999999998
    1.4130027920707082
    0.15367976585756082
    0.46142633246812403
    -1.620647789230658
    -1.3540178492487063
    -1.2310268787102738
    -0.9490372374384459

    >>> kurtosis = river.stats.Kurtosis(bias=True)
    >>> for x in X:
    ...     print(kurtosis.update(x).get())
    -3
    -2.0
    -1.5
    -1.011599627723906
    -0.9615800585356089
    -0.6989395431537853
    -1.4252699121794408
    -1.311437071070812
    -1.246289111322894
    -1.082283689864171

    >>> for i in range(1, len(X)+1):
    ...     print(scipy.stats.kurtosis(X[:i], bias=True))
    -3.0
    -2.0
    -1.4999999999999998
    -1.0115996277239057
    -0.9615800585356098
    -0.6989395431537861
    -1.425269912179441
    -1.3114370710708125
    -1.2462891113228936
    -1.0822836898641714

    References
    ----------
    [^1]: [Wikipedia article on algorithms for calculating variance](https://www.wikiwand.com/en/Algorithms_for_calculating_variance#/Covariance)

    """

    def __init__(self, bias=False):
        super().__init__()
        self.bias = bias

    def update(self, x):
        self.count.update()
        self._update_delta(x)
        self._update_m1(x)
        self._update_sum_delta()
        self._update_m4()
        self._update_m3()
        self._update_m2()
        return self

    def get(self):
        n = self.count.get()
        kurtosis = n * self.M4 / self.M2 ** 2 if self.M2 != 0 else 0
        if not self.bias and n > 3:
            return 1 / (n - 2) / (n - 3) * ((n ** 2 - 1) * kurtosis - 3 * (n - 1) ** 2)
        return kurtosis - 3
