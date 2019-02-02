import math

from . import _moments


class Skew(_moments.CentralMoments):
    """Computes a running skew using Welford's algorithm.

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

        >>> skew = creme.stats.Skew()
        >>> pprint.pprint([skew.update(v).get() for v in array])
        [0,
         0.0,
         -0.6043053732501439,
         0.2960327239981376,
         0.5234724473423674,
         0.7712778043924866,
         0.39022088752624845,
         0.278892645224261,
         0.37425953513864063,
         0.3476878073823696]

        >>> pprint.pprint(scipy.stats.skew(array))
        0.34768780738236926

    References:
        - `Welford's online algorithm <https://www.wikiwand.com/en/Algorithms_for_calculating_variance#/Welford's_Online_algorithm>`_

    """

    @property
    def name(self):
        return 'skew'

    def update(self, x):
        self.count.update()
        self._update_delta(x)
        self._update_m1(x)
        self._update_sum_delta()
        self._update_m3()
        self._update_m2()
        return self

    def get(self):
        return (math.sqrt(self.count.get()) * self.M3) / self.M2**(3 / 2) if self.M2 != 0 else 0
