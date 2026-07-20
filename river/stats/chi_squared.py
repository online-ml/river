from __future__ import annotations

import collections
import typing

from river import stats


class ChiSquared(stats.base.Bivariate):
    """Streaming Chi-squared statistic.

    Maintains a contingency table between two variables `x` and `y` and computes the
    Chi-squared statistic incrementally. This can be used to measure the dependency
    between two categorical variables in a streaming setting.

    Examples
    --------
    >>> from river import stats

    >>> chi2 = stats.ChiSquared()

    >>> data = [
    ...     ("A", 0),
    ...     ("A", 0),
    ...     ("B", 1),
    ...     ("B", 1),
    ... ]

    >>> for x, y in data:
    ...     chi2.update(x, y)

    >>> round(chi2.get(), 3)
    4.0

    A rolling version can be obtained by wrapping with `utils.Rolling`:

    >>> from river import utils

    >>> data = [
    ...     ("A", 0),
    ...     ("A", 0),
    ...     ("B", 1),
    ...     ("B", 1),
    ...     ("C", 0),
    ... ]

    >>> chi2 = utils.Rolling(stats.ChiSquared, window_size=4)

    >>> for x, y in data:
    ...     chi2.update(x, y)

    >>> round(chi2.get(), 3)
    4.0

    """

    def __init__(self) -> None:
        self.counts: collections.defaultdict[typing.Any, collections.Counter[typing.Any]] = (
            collections.defaultdict(collections.Counter)
        )
        self.x_totals: collections.Counter[typing.Any] = collections.Counter()
        self.y_totals: collections.Counter[typing.Any] = collections.Counter()
        self.n: int = 0

    @property
    def name(self) -> str:
        return "chi_squared"

    # TODO
    def update(self, x: typing.Any, y: typing.Any) -> None:
        self.counts[x][y] += 1
        self.x_totals[x] += 1
        self.y_totals[y] += 1
        self.n += 1

    # TODO
    def revert(self, x: typing.Any, y: typing.Any) -> None:
        self.counts[x][y] -= 1
        if self.counts[x][y] <= 0:
            del self.counts[x][y]
        if not self.counts[x]:
            del self.counts[x]

        self.x_totals[x] -= 1
        if self.x_totals[x] <= 0:
            del self.x_totals[x]

        self.y_totals[y] -= 1
        if self.y_totals[y] <= 0:
            del self.y_totals[y]

        self.n -= 1

    @property
    def degrees_of_freedom(self) -> int:
        """Return the degrees of freedom of the contingency table."""
        r = len(self.x_totals)
        c = len(self.y_totals)
        if r <= 1 or c <= 1:
            return 0
        return (r - 1) * (c - 1)

    @property
    def p_value(self) -> float:
        """Return the p-value associated with the Chi-squared statistic."""
        import scipy.stats

        df = self.degrees_of_freedom
        if df <= 0:
            return 1.0
        return float(scipy.stats.chi2.sf(self.get(), df))

    def get(self) -> float:
        if self.n == 0:
            return 0.0

        chi2 = 0.0
        for x in self.counts:
            for y in self.y_totals:
                observed = self.counts[x].get(y, 0)
                expected = (self.x_totals[x] * self.y_totals[y]) / self.n
                if expected > 0:
                    chi2 += (observed - expected) ** 2 / expected

        return chi2
