from __future__ import annotations

import bisect
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
    ...     hist.update(x)

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
        # Operate on the underlying list directly: going through UserList's
        # __getitem__/__len__ (with their isinstance checks) dominates the cost
        # of this hot path.
        data = self.data

        # Insert the bin if the histogram is empty
        if not data:
            data.append(Bin(x, x, 1))
            return

        # Bisect to find the left-most bin whose right edge is >= x. We inline
        # the comparison (rather than using the bisect module or Bin.__lt__) to
        # avoid per-iteration function-call overhead.
        lo = 0
        hi = len(data)
        while lo < hi:
            mid = (lo + hi) // 2
            if data[mid].right < x:
                lo = mid + 1
            else:
                hi = mid

        if lo == len(data):
            # x is past the right-most bin
            data.append(Bin(x, x, 1))
        # Increment the bin counter if x is part of the lo-th bin
        elif x >= data[lo].left:
            data[lo].count += 1
        # Insert the bin if it is between bin lo-1 and bin lo
        else:
            data.insert(lo, Bin(x, x, 1))

        # Bins have to be merged if there are more than max_bins
        if len(data) > self.max_bins:
            self._shrink(1)

    def _shrink(self, k):
        """Shrinks the histogram by merging the k closest pairs of bins."""

        data = self.data

        if k == 1:
            # Find the closest pair of bins in a single pass over the right edges
            min_diff = math.inf
            min_idx = 0
            prev_right = data[0].right
            for idx in range(1, len(data)):
                right = data[idx].right
                diff = right - prev_right
                if diff < min_diff:
                    min_diff = diff
                    min_idx = idx - 1
                prev_right = right

            # Merge the bins
            data[min_idx] += data.pop(min_idx + 1)  # Calls Bin.__iadd__
            return

        if k < 1:
            return

        indexes = range(len(data) - 1)

        def bin_distance(i):
            return data[i + 1].right - data[i].right

        for i in sorted(heapq.nsmallest(n=k, iterable=indexes, key=bin_distance), reverse=True):
            data[i] += data.pop(i + 1)  # Calls Bin.__iadd__

    def iter_cdf(self, X, verbose=False):
        """Yields CDF values for a sorted iterable of values.

        This is faster than calling `cdf` with many values.

        Examples
        --------

        >>> from river import sketch

        >>> hist = sketch.Histogram()
        >>> for x in range(4):
        ...     hist.update(x)

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

        >>> empty = sketch.Histogram()
        >>> list(empty.iter_cdf([0, 1]))
        [0.0, 0.0]

        """

        # Nothing has been seen yet: the CDF is identically zero.
        if not self.n:
            for _ in X:
                yield 0.0
            return

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
        ...     hist.update(x)

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
        """Merge two histograms, conserving the total count.

        Interval bins are spread proportionally over the tiles defined by the
        union of both histograms' edges, while point bins (``left == right``)
        contribute their whole count to a single tile. The result is then shrunk
        back down to ``max_bins``.

        Examples
        --------

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

        Counts are conserved, including for point-valued histograms:

        >>> a, b = Histogram(), Histogram()
        >>> for x in range(5):
        ...     a.update(x)
        >>> for x in range(2, 7):
        ...     b.update(x)
        >>> merged = a + b
        >>> merged.n
        10
        >>> sum(bin.count for bin in merged)
        10.0
        >>> merged.cdf(10)
        1.0

        """

        merged = Histogram(max_bins=max(self.max_bins, other.max_bins))
        merged.n = self.n + other.n

        bins = list(itertools.chain(self, other))
        if not bins:
            return merged

        # The output bins tile the union of both edge sets without overlapping.
        edges = sorted({e for b in bins for e in (b.left, b.right)})
        if len(edges) == 1:
            merged.append(Bin(edges[0], edges[0], sum(b.count for b in bins)))
            return merged

        tiles = [Bin(lo, hi, 0.0) for lo, hi in zip(edges, edges[1:])]

        for b in bins:
            if b.left == b.right:
                # Point mass: assign the whole count to a single tile so that it
                # is never counted twice on a shared edge.
                j = bisect.bisect_left(edges, b.left)
                j = 0 if j == 0 else j - 1
                tiles[j].count += b.count
            else:
                # Spread the count proportionally over the overlapping tiles.
                start = bisect.bisect_left(edges, b.left)
                for tile in itertools.islice(tiles, start, None):
                    if tile.left >= b.right:
                        break
                    tile.count += b.count * coverage_ratio(tile, b)

        merged.data.extend(tile for tile in tiles if tile.count)
        merged._shrink(k=len(merged) - merged.max_bins)

        return merged

    def __repr__(self):
        return "\n".join(str(b) for b in self)

    def __str__(self):
        return repr(self)
