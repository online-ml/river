import bisect
import collections
import math


__all__ = ['Histogram']


class Bin:
    """A Bin is an element of a Histogram."""

    def __init__(self, right, count=1):
        self.right = right
        self.count = count

    def __lt__(self, other):
        return self.right < other.right

    def __le__(self, other):
        return self.right <= other.right

    def __add__(self, other):
        total = self.count + other.count
        return Bin(
            right=(self.right * self.count + other.right * other.count) / total,
            count=total
        )


class Histogram(collections.UserList):
    """Streaming histogram.

    Parameters:
        max_bins (int): Maximal number of bins.

    Attributes:
        n (int): Total number of seen values.

    Example:

        ::

            >>> from creme import utils
            >>> import matplotlib.pyplot as plt
            >>> import numpy as np

            >>> np.random.seed(42)

            >>> values = np.hstack((
            ...     np.random.normal(-3, 1, 1000),
            ...     np.random.normal(3, 1, 1000),
            ... ))

            >>> hist = utils.Histogram(max_bins=80)

            >>> for x in values:
            ...     hist = hist.update(x)

            >>> ax = plt.vlines(
            ...     x=[h.right for h in hist],
            ...     ymin=0,
            ...     ymax=[h.count for h in hist]
            ... )

        .. image:: ../_static/histogram_docstring.svg
            :align: center

    References:
        1. `A Streaming Parallel Decision Tree Algorithm <http://jmlr.org/papers/volume11/ben-haim10a/ben-haim10a.pdf>`_
        2. `Go implementation <https://github.com/VividCortex/gohistogram>`_

    """

    def __init__(self, max_bins=256):
        super().__init__()
        self.max_bins = max_bins
        self.min = math.inf
        self.n = 0

    def update(self, x):

        self.n += 1
        self.min = min(self.min, x)

        # Insert the value
        bisect.insort_left(self, Bin(x))

        # Merge the two closest bins if there are more than max_bins
        if len(self) > self.max_bins:

            # Find the closest pair of bins
            min_diff = math.inf
            min_idx = None
            for idx, (b1, b2) in enumerate(zip(self[:-1], self[1:])):
                diff = b2.right - b1.right
                if diff < min_diff:
                    min_diff = diff
                    min_idx = idx

            # Merge the bins
            self[min_idx] += self.pop(min_idx + 1)

        return self

    def cdf(self, x):
        """Cumulative distribution function.

        Example:

            ::

                >>> from creme import utils

                >>> hist = Histogram()
                >>> for x in range(4):
                ...     hist = hist.update(x)

                >>> print(hist)
                [0.00000, 0.00000]: 1
                (0.00000, 1.00000]: 1
                (1.00000, 2.00000]: 1
                (2.00000, 3.00000]: 1

                >>> hist.cdf(0)
                0.25

                >>> hist.cdf(.5)
                0.375

                >>> hist.cdf(1)
                0.5

                >>> hist.cdf(2.5)
                0.875

        """

        # Handle edge cases
        if not self or x < self.min:
            return 0.
        elif x >= self[-1].right:
            return 1.

        c = 0

        # Handle the first bin
        b = self[0]
        if x < b.right:
            c += b.count * (x - self.min) / (b.right - self.min)
            return c / self.n
        c += b.count

        # Handle the rest of the bins
        for b1, b2 in zip(self[:-1], self[1:]):
            if x < b2.right:
                # Interpolate
                c += b2.count * (x - b1.right) / (b2.right - b1.right)
                break
            c += b2.count

        return c / self.n

    def __str__(self):
        return (
            f'[{self.min:.5f}, {self[0].right:.5f}]: {self[0].count}\n' +
            '\n'.join(
                f'({b1.right:.5f}, {b2.right:.5f}]: {b1.count}'
                for b1, b2 in zip(self[:-1], self[1:])
            )
        )
