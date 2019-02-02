from . import _moments


class Kurtosis(_moments.CentralMoments):
    """Computes a running kurtosis using Welford's algorithm.

    Attributes:
        central_moments (stats.CentralMoments)

    Example:

    ::

        >>> import creme
        >>> import pprint
        >>> import scipy.stats
        >>> import numpy as np

        >>> np.random.seed(42)
        >>> array = np.random.normal(loc = 0, scale = 1, size = 10)

        >>> kurtosis = creme.stats.Kurtosis()
        >>> pprint.pprint([kurtosis.update(v).get() for v in array])
        [-3,
         -2.0,
         -1.5,
         -1.011599627723906,
         -0.9615800585356089,
         -0.6989395431537853,
         -1.4252699121794408,
         -1.311437071070812,
         -1.246289111322894,
         -1.082283689864171]

        >>> pprint.pprint(scipy.stats.kurtosis(array))
        -1.0822836898641714

    References:
        - `Welford's online algorithm <https://www.wikiwand.com/en/Algorithms_for_calculating_variance#/Welford's_Online_algorithm>`_

    """

    @property
    def name(self):
        return 'kurtosis'

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
        return (self.count.get() * self.M4) / self.M2**2 - 3 if self.M2 != 0 else -3
