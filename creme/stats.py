"""
Module for computing running statistics
"""
import abc
import math


__all__ = [
    'Count',
    'Kurtosis',
    'Max',
    'Mean',
    'Min',
    'PeakToPeak',
    'Skew',
    'SmoothMean',
    'Sum',
    'Variance'
]


class RunningStatistic(abc.ABC):

    @property
    @abc.abstractmethod
    def name(self):
        pass

    @abc.abstractmethod
    def update(self):
        pass

    @abc.abstractmethod
    def get(self) -> float:
        pass


class Count(RunningStatistic):
    """Simply counts the number of times ``update`` is called."""

    def __init__(self):
        super().__init__()
        self.n = 0

    @property
    def name(self):
        return 'count'

    def update(self, x=None):
        self.n += 1
        return self

    def get(self):
        return self.n


class Mean(RunningStatistic):
    """Computes a running mean.

    Attributes:
        count (stats.Count)
        mu (float): The current estimated mean.

    """

    def __init__(self):
        super().__init__()
        self.count = Count()
        self.mu = 0

    @property
    def name(self):
        return 'mean'

    def update(self, x):
        self.mu += (x - self.mu) / self.count.update().get()
        return self

    def get(self):
        return self.mu


class SmoothMean(Mean):
    """Computes a running mean using additive smoothing.

    Parameters:
        prior (float): Prior mean.
        prior_weight (float): Strengh of the prior mean.

    Attributes:
        count (stats.Count)
        mu (float): The current estimated mean.

    References:

    - `Additive smoothing <https://www.wikiwand.com/en/Additive_smoothing>`_

    """

    def __init__(self, prior, prior_weight):
        super().__init__()
        self.prior = prior
        self.prior_weight = prior_weight

    @property
    def name(self):
        return 'smooth_mean'

    def get(self):
        numerator = self.mu * self.count.get() + self.prior * self.prior_weight
        denominator = self.count.get() + self.prior_weight
        return numerator / denominator


class Variance(RunningStatistic):
    """Computes a running variance using Welford's algorithm.

    Attributes:
        mean (stats.Mean)
        sos (float): The running sum of squares.

    References:

    - `Welford's online algorithm <https://www.wikiwand.com/en/Algorithms_for_calculating_variance#/Welford's_Online_algorithm>`_

    """

    def __init__(self):
        super().__init__()
        self.mean = Mean()
        self.sos = 0

    @property
    def name(self):
        return 'variance'

    def update(self, x):
        old_mean = self.mean.get()
        new_mean = self.mean.update(x).get()
        self.sos += (x - old_mean) * (x - new_mean)
        return self

    def get(self):
        return self.sos / self.mean.count.n if self.sos else 0


class Max(RunningStatistic):
    """Computes a running max.

    Attributes:
        max : The running max.

    """

    def __init__(self):
        super().__init__()
        self.max = - math.inf

    @property
    def name(self):
        return 'max'

    def update(self, x):
        if x > self.max:
            self.max = x

        return self

    def get(self):
        return self.max


class Min(RunningStatistic):
    """Computes a running min.

    Attributes:
        min : The running min.

    """

    def __init__(self):
        super().__init__()
        self.min = math.inf

    @property
    def name(self):
        return 'min'

    def update(self, x):
        if x < self.min:
            self.min = x

        return self

    def get(self):
        return self.min


class PeakToPeak(RunningStatistic):
    """Computes a running peak to peak (max - min).

    Attributes:
        max (stats.Max)
        min (stats.Min)
        p2p (float): The running peak to peak.

    """

    def __init__(self):
        super().__init__()
        self.max = Max()
        self.min = Min()

    @property
    def name(self):
        return 'p2p'

    def update(self, x):
        self.max.update(x)
        self.min.update(x)

        return self

    def get(self):
        return self.max.get() - self.min.get()


class Sum(RunningStatistic):
    """Computes a running Sum.

    Attributes:
        sum (float) : The running sum.

    """

    def __init__(self):
        super().__init__()
        self.sum = 0.

    @property
    def name(self):
        return 'sum'

    def update(self, x):
        self.sum += x
        return self

    def get(self):
        return self.sum


class CentralMoments(RunningStatistic):
    """Computes central moments using Welford's algorithm.

    Attributes:
        count (stats.Count)
        delta (float): Mean of differences.
        sum_delta (float): Mean of sum of differences.
        M1 (float): sums of powers of differences from the mean order 1.
        M2 (float): sums of powers of differences from the mean order 2.
        M3 (float): sums of powers of differences from the mean order 3.
        M4 (float): sums of powers of differences from the mean order 4.

    References:
        - `Welford's online algorithm <https://www.wikiwand.com/en/Algorithms_for_calculating_variance#/Welford's_Online_algorithm>`_
    """

    def __init__(self):
        self.count = Count()

        self.delta = 0
        self.sum_delta = 0

        self.M1 = 0
        self.M2 = 0
        self.M3 = 0
        self.M4 = 0

    def _update_delta(self, x):
        self.delta = (x - self.sum_delta) / self.count.get()
        return self

    def _update_sum_delta(self):
        self.sum_delta += self.delta
        return self

    def _update_m1(self, x):
        self.M1 = (x - self.sum_delta) * self.delta * (self.count.get() - 1)
        return self

    def _update_m2(self):
        self.M2 += self.M1
        return self

    def _update_m3(self):
        self.M3 += (self.M1 * self.delta * (self.count.get() - 2) - 3 *
                    self.delta * self.M2)
        return self

    def _update_m4(self):
        delta_square = self.delta ** 2
        self.M4 += (self.M1 * delta_square *
                    (self.count.get() ** 2 - 3 * self.count.get() + 3) +
                    6 * delta_square * self.M2 - 4 * self.delta * self.M3)
        return self


class Kurtosis(CentralMoments):
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


class Skew(CentralMoments):
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
