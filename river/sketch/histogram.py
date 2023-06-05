from __future__ import annotations

import collections
import heapq
import itertools
import math

from river import base

__all__ = ["Histogram"]


class Bin:
    """A Bin is an element of a Histogram."""

    __slots__ = ["left", "right", "count"]

    def __init__(self, left, right, count):
        self.left = left
        self.right = right
        self.count = count

    def __iadd__(self, other):
        """Merge with another bin."""
        if other.left < self.left:
            self.left = other.left
        if other.right > self.right:
            self.right = other.right
        self.count += other.count
        return self

    def __lt__(self, other):
        return self.right < other.left

    def __eq__(self, other):
        return self.left == other.left and self.right == other.right

    def __repr__(self):
        return f"[{self.left:.5f}, {self.right:.5f}]: {self.count}"


def coverage_ratio(x: Bin, y: Bin) -> float:
    """Returns the amount of y covered by x.

    Examples
    --------

    >>> coverage_ratio(Bin(1, 2, 0), Bin(1, 2, 0))
    1.0

    >>> coverage_ratio(Bin(1, 3, 0), Bin(2, 4, 0))
    0.5

    >>> coverage_ratio(Bin(1, 3, 0), Bin(3, 5, 0))
    0.0

    >>> coverage_ratio(Bin(1, 3, 0), Bin(0, 4, 0))
    0.5

    >>> coverage_ratio(Bin(0, 4, 0), Bin(1, 3, 0))
    1.0

    >>> coverage_ratio(Bin(1, 3, 0), Bin(0, 1, 0))
    0.0

    >>> coverage_ratio(Bin(1, 1, 0), Bin(1, 1, 0))
    1.0

    """
    if y.left == y.right:
        return float(x.left <= y.left <= x.right)
    return max(0, min(x.right, y.right) - max(x.left, y.left)) / (y.right - y.left)


class Histogram(collections.UserList, base.Base):
    """Streaming histogram.

    Parameters
    ----------
    max_bins
        Maximal number of bins.

    Attributes
    ----------
    n
        Total number of seen values.

    Examples
    --------

    >>> from river import sketch
    >>> import numpy as np

    >>> np.random.seed(42)

    >>> values = np.hstack((
    ...     np.random.normal(-3, 1, 1000),
    ...     np.random.normal(3, 1, 1000),
    ... ))

    >>> hist = sketch.Histogram(max_bins=15)

    >>> for x in values:
    ...     hist = hist.update(x)

    >>> for bin in hist:
    ...     print(bin)
    [-6.24127, -6.24127]: 1
    [-5.69689, -5.19881]: 8
    [-5.12390, -4.43014]: 57
    [-4.42475, -3.72574]: 158
    [-3.71984, -3.01642]: 262
    [-3.01350, -2.50668]: 206
    [-2.50329, -0.81020]: 294
    [-0.80954, 0.29677]: 19
    [0.40896, 0.82733]: 7
    [0.84661, 1.25147]: 24
    [1.26029, 2.30758]: 178
    [2.31081, 3.05701]: 284
    [3.05963, 3.69695]: 242
    [3.69822, 5.64434]: 258
    [6.13775, 6.19311]: 2

    References
    ----------
    [^1]: [Ben-Haim, Y. and Tom-Tov, E., 2010. A streaming parallel decision tree algorithm. Journal of Machine Learning Research, 11(Feb), pp.849-872.](http://jmlr.org/papers/volume11/ben-haim10a/ben-haim10a.pdf)
    [^2]: [Go implementation](https://github.com/VividCortex/gohistogram)

    """

    def __init__(self, max_bins=256):
        super().__init__()
        self.max_bins = max_bins
        self.n = 0

    def update(self, x):
        self.n += 1
        b = Bin(x, x, 1)

        # Insert the bin if the histogram is empty
        if not self:
            self.append(b)
            return self

        # Use bisection to find where to insert
        # We don't use the bisect module in order to save some CPU cycles
        lo = 0
        hi = len(self)
        i = (lo + hi) // 2
        while lo < hi:
            if self[i] < b:
                lo = i + 1
            else:
                hi = i
            i = (lo + hi) // 2

        if i == len(self):
            # x is past the right-most bin
            self.append(b)
        else:
            # Increment the bin counter if x is part of the ith bin
            if x >= self[i].left:
                self[i].count += 1
            # Insert the bin if it is between bin i-1 and bin i
            else:
                self.insert(i, b)

        # Bins have to be merged if there are more than max_bins
        if len(self) > self.max_bins:
            self._shrink(1)

        return self

    def _shrink(self, k):
        """Shrinks the histogram by merging the two closest bins."""

        if k == 1:
            # Find the closest pair of bins
            min_diff = math.inf
            min_idx = None
            for idx, (b1, b2) in enumerate(zip(self.data[:-1], self.data[1:])):
                diff = b2.right - b1.right
                if diff < min_diff:
                    min_diff = diff
                    min_idx = idx

            # Merge the bins
            self[min_idx] += self.pop(min_idx + 1)
            return

        indexes = range(len(self) - 1)

        def bin_distance(i):
            return self[i + 1].right - self[i].right

        for i in sorted(heapq.nsmallest(n=k, iterable=indexes, key=bin_distance), reverse=True):
            self[i] += self.pop(i + 1)  # Calls Bin.__iadd__

    def iter_cdf(self, X, verbose=False):
        """Yields CDF values for a sorted iterable of values.

        This is faster than calling `cdf` with many values.

        Examples
        --------

        >>> from river import sketch

        >>> hist = sketch.Histogram()
        >>> for x in range(4):
        ...     hist = hist.update(x)

        >>> print(hist)
        [0.00000, 0.00000]: 1
        [1.00000, 1.00000]: 1
        [2.00000, 2.00000]: 1
        [3.00000, 3.00000]: 1

        >>> X = [-1, 0, .5, 1, 2.5, 3.5]
        >>> for x, cdf in zip(X, hist.iter_cdf(X)):
        ...     print(x, cdf)
        -1 0.0
        0 0.25
        0.5 0.25
        1 0.5
        2.5 0.75
        3.5 1.0

        """

        bins = iter(self)
        b = next(bins)
        INF = Bin(math.inf, math.inf, 0)

        cdf = 0

        for x in X:
            while x >= b.right:
                cdf += b.count
                b = next(bins, INF)

            if x > b.left:
                yield (cdf + b.count * (x - b.left) / (b.right - b.left)) / self.n
                continue

            yield cdf / self.n

    def cdf(self, x):
        """Cumulative distribution function.

        Examples
        --------

        >>> from river import sketch

        >>> hist = sketch.Histogram()
        >>> for x in range(4):
        ...     hist = hist.update(x)

        >>> print(hist)
        [0.00000, 0.00000]: 1
        [1.00000, 1.00000]: 1
        [2.00000, 2.00000]: 1
        [3.00000, 3.00000]: 1

        >>> hist.cdf(-1)
        0.0

        >>> hist.cdf(0)
        0.25

        >>> hist.cdf(.5)
        0.25

        >>> hist.cdf(1)
        0.5

        >>> hist.cdf(2.5)
        0.75

        >>> hist.cdf(3.5)
        1.0

        """
        return next(self.iter_cdf([x]))

    def __add__(self, other):
        """

        Example:

            >>> h1 = Histogram()
            >>> for b in [Bin(0, 2, 4), Bin(4, 5, 9)]:
            ...     h1.append(b)

            >>> h2 = Histogram()
            >>> for b in [Bin(1, 3, 8), Bin(3, 4, 5)]:
            ...     h2.append(b)

            >>> h1 + h2
            [0.00000, 1.00000]: 2.0
            [1.00000, 2.00000]: 6.0
            [2.00000, 3.00000]: 4.0
            [3.00000, 4.00000]: 5.0
            [4.00000, 5.00000]: 9.0

        """

        xs = iter(self)
        ys = iter(other)
        rights = heapq.merge(
            itertools.chain.from_iterable((b.left, b.right) for b in self),
            itertools.chain.from_iterable((b.left, b.right) for b in other),
        )

        b = Bin(next(rights), next(rights), 0)

        x = next(xs)
        y = next(ys)

        hist = Histogram(max_bins=max(self.max_bins, other.max_bins))

        while True:
            b.count += coverage_ratio(b, x) * x.count
            b.count += coverage_ratio(b, y) * y.count

            if b.count:
                hist.append(b)

            try:
                b = Bin(b.right, next(rights), 0)
            except StopIteration:
                break

            if b.left >= x.right:
                x = next(xs, x)
            if b.left >= y.right:
                y = next(ys, y)

        hist._shrink(k=len(hist) - hist.max_bins)

        return hist

    def __repr__(self):
        return "\n".join(str(b) for b in self)

    def __str__(self):
        return repr(self)
