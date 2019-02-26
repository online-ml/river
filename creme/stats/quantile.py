import math

from . import base
from . import _sorted_window


class Quantile(base.RunningStatistic):
    """Compute the running quantile. Default value compute median.

    Uses the P-square algorithm to calculate the quantile. The code is inspired by
    LiveStat's implementation [2].

    Attributes:
        quantile (float): quantile you want compute the value
        must be between 0 and 1 excluded.

    Example:

    ::

        >>> from creme import stats
        >>> import numpy as np

        >>> np.random.seed(42*1337)
        >>> mu, sigma = 0, 1
        >>> s = np.random.normal(mu, sigma, 1337*42)

        >>> median = stats.Quantile(0.5)
        >>> for x in s:
        ...    _ = median.update(x)

        >>> print(f'The estimated value of the 50th (median) quantile is {median.get():.4f}')
        The estimated value of the 50th (median) quantile is -0.0006
        >>> print(f'The real value of the 50th (median) quantile is {np.median(s):.4f}')
        The real value of the 50th (median) quantile is -0.0002

        >>> percentile_17 = stats.Quantile(0.17)
        >>> for x in s:
        ...    _ = percentile_17.update(x)

        >>> print(f'The estimated value of the 17th quantile is  {percentile_17.get():.4f}')
        The estimated value of the 17th quantile is  -0.9522
        >>> print(f'The real value of the 17th quantile is {np.percentile(s,17):.4f}')
        The real value of the 17th quantile is -0.9510

    References:

    1. `The P2 Algorithm for Dynamic Statistical Computing Calculation of Quantiles and Editor Histograms Without Storing Observations  <https://www.cse.wustl.edu/~jain/papers/ftp/psqr.pdf>`_
    2. `Python implementation <https://github.com/cxxr/LiveStats/blob/master/livestats/livestats.py>`_

    """

    def __init__(self, quantile=0.5):

        if 0 < quantile < 1:
            self.quantile = quantile
        else:
            raise ValueError('quantile must be between 0 and 1 excluded')

        self.desired_marker_position = [
            0,
            self.quantile / 2,
            self.quantile,
            (1 + self.quantile) / 2,
            1
        ]
        self.marker_position = [
            1,
            1 + 2 * self.quantile,
            1 + 4 * self.quantile,
            3 + 2 * self.quantile,
            5
        ]
        self.position = list(range(1, 6))
        self.heights = []
        self.heights_sorted = False

    @property
    def name(self):
        return 'quantile'

    def _find_k(self, x):

        if x < self.heights[0]:
            self.heights[0] = x
            k = 1

        else:
            for i in range(1, 5):
                if self.heights[i - 1] <= x and x < self.heights[i]:
                    k = i
                    break
            else:
                k = 4
                if self.heights[-1] < x:
                    self.heights[-1] = x
        return k

    @classmethod
    def _compute_P2(cls, qp1, q, qm1, d, np1, n, nm1):

        d = float(d)
        n = float(n)
        np1 = float(np1)
        nm1 = float(nm1)

        outer = d / (np1 - nm1)
        inner_left = (n - nm1 + d) * (qp1 - q) / (np1 - n)
        inner_right = (np1 - n - d) * (q - qm1) / (n - nm1)

        return q + outer * (inner_left + inner_right)

    def _adjust(self):

        for i in range(1, 4):
            n = self.position[i]
            q = self.heights[i]

            d = self.marker_position[i] - n

            if (d >= 1 and self.position[i + 1] - n > 1) or (d <= -1 and self.position[i - 1] - n < -1):
                d = int(math.copysign(1, d))

                qp1 = self.heights[i + 1]
                qm1 = self.heights[i - 1]
                np1 = self.position[i + 1]
                nm1 = self.position[i - 1]

                qn = self._compute_P2(qp1, q, qm1, d, np1, n, nm1)

                if qm1 < qn and qn < qp1:
                    self.heights[i] = qn
                else:
                    self.heights[i] = q + d * (self.heights[i + d] - q) / (self.position[i + d] - n)

                self.position[i] = n + d

        return self

    def update(self, x):

        # Initialisation
        if len(self.heights) != 5:
            self.heights.append(x)

        else:
            if not self.heights_sorted:
                self.heights.sort()
                self.heights_sorted = True

            # Find cell k such that qk < Xj <= qk+i and adjust extreme values (q1 and q) if necessary
            k = self._find_k(x)

            # Increment all positions greater than k
            self.position = [j if i < k else j + 1 for i, j in enumerate(self.position)]
            self.marker_position = [
                x + y
                for x, y in zip(self.marker_position, self.desired_marker_position)
            ]

            # Adjust heights of markers 2-4 if necessary
            self._adjust()

        return self

    def get(self):
        if self.heights_sorted:
            return self.heights[2]
        else:
            self.heights.sort()
            length = len(self.heights)
            return self.heights[int(min(max(length - 1, 0), length * self.quantile))]

class RollingQuantile(base.RunningStatistic):
    """Calculate the rolling quantile with a given window size.

    Attributes:
        quantile (float): quantile you want compute the value
            must be between 0 and 1 excluded.
        window_size (int): size of the window to compute the rolling quantile.
        current_window (collections.deque): Store values that are in the current window.

    Example:

    ::

        >>> from creme import stats
        >>> import numpy as np

        >>> rolling_quantile = stats.RollingQuantile(window_size = 100,
        ...                                          quantile = 0.5)

        >>> for i in range(0,1001):
        ...     _ = rolling_quantile.update(i)
        ...     if i%100 == 0:
        ...         print(rolling_quantile.get())
        0
        50
        150
        250
        350
        450
        550
        650
        750
        850
        950

    References:

    - `Left sorted <https://stackoverflow.com/questions/8024571/insert-an-item-into-sorted-list-in-python>`_


    """

    def __init__(self, window_size, quantile=0.5):

        if 0 < quantile < 1:
            self.quantile = quantile
        else:
            raise ValueError('quantile must be between 0 and 1 excluded')

        self.quantile = quantile
        self.window_size = window_size
        self.sorted_window = _sorted_window._SortedWindow(window_size=self.window_size)

        self.idx_percentile = int(round(self.quantile * self.window_size + 0.5)) - 1

    @property
    def name(self):
        return 'rolling_quantile'

    def update(self, x):
        # Update current window.
        self.sorted_window.update(x)

        return self

    def get(self):
        if len(self.sorted_window.get()) < self.window_size:
            _idx_percentile = int(round(self.quantile * len(self.sorted_window.get()) + 0.5)) - 1
            return self.sorted_window.get()[_idx_percentile]

        return self.sorted_window.get()[self.idx_percentile]
