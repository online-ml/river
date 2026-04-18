from __future__ import annotations

import collections
# import math

from river import stats


class ChiSquared(stats.base.Bivariate):
    """Streaming Chi-squared statistic.

    Maintains a contingency table between feature values (x) and classes (y)
    and computes the Chi-squared statistic incrementally.

    This can be used to measure the dependency between a categorical feature
    and a target variable in a streaming setting.

    Examples
    --------
    >>> from river import stats

    >>> chi = stats.ChiSquared()

    >>> data = [
    ...     ("A", 0),
    ...     ("A", 0),
    ...     ("B", 1),
    ...     ("B", 1),
    ... ]

    >>> for x, y in data:
    ...     chi.update(x, y)

    >>> round(chi.get(), 3)
    4.0
    """

    def __init__(self):
        # counts[value][class] = frequency
        self.counts = collections.defaultdict(collections.Counter)

        # total counts per class
        self.class_totals = collections.Counter()

        # total counts per feature value
        self.value_totals = collections.Counter()

        # total observations
        self.n = 0

    @property
    def name(self):
        return "chi_squared"

    def update(self, x, y):
        """Update the statistic with a new observation.

        Parameters
        ----------
        x
            Feature value (categorical).
        y
            Target class.
        """
        self.counts[x][y] += 1
        self.class_totals[y] += 1
        self.value_totals[x] += 1
        self.n += 1

    def get(self):
        """Return the current Chi-squared statistic."""
        if self.n == 0:
            return 0.0

        chi2 = 0.0

        # Iterate over contingency table
        for x in self.counts:
            for y in self.class_totals:

                observed = self.counts[x].get(y, 0)

                expected = (
                    self.value_totals[x] * self.class_totals[y]
                ) / self.n

                if expected > 0:
                    chi2 += (observed - expected) ** 2 / expected

        return chi2

    