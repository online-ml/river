from __future__ import annotations

import collections

from river import stats, utils


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

    >>> print(round(chi.p_value, 3))
    0.046

    >>> chi.is_significant(alpha=0.05)
    True

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

    def revert(self, x, y):
        """Revert the statistic with a new observation.

        Parameters
        ----------
        x
            Feature value (categorical).
        y
            Target class.
        """
        self.counts[x][y] -= 1
        if self.counts[x][y] <= 0:
            del self.counts[x][y]
        if not self.counts[x]:
            del self.counts[x]

        self.class_totals[y] -= 1
        if self.class_totals[y] <= 0:
            del self.class_totals[y]

        self.value_totals[x] -= 1
        if self.value_totals[x] <= 0:
            del self.value_totals[x]

        self.n -= 1

    @property
    def degrees_of_freedom(self):
        """Return the degrees of freedom of the contingency table."""
        r = len(self.value_totals)
        c = len(self.class_totals)
        if r <= 1 or c <= 1:
            return 0
        return (r - 1) * (c - 1)

    @property
    def p_value(self):
        """Return the p-value associated with the Chi-squared statistic."""
        import scipy.stats

        df = self.degrees_of_freedom
        if df <= 0:
            return 1.0
        return float(scipy.stats.chi2.sf(self.get(), df))

    def is_significant(self, alpha=0.05):
        """Return whether the test is significant at a given alpha level."""
        return bool(self.p_value < alpha)

    def get(self):
        """Return the current Chi-squared statistic."""
        if self.n == 0:
            return 0.0

        chi2 = 0.0

        # Iterate over contingency table
        for x in self.counts:
            for y in self.class_totals:
                observed = self.counts[x].get(y, 0)
                expected = (self.value_totals[x] * self.class_totals[y]) / self.n
                if expected > 0:
                    chi2 += (observed - expected) ** 2 / expected

        return chi2


class RollingChiSquared(stats.base.RollingBivariate, utils.Rolling):
    """Running Chi-squared statistic over a window.

    Maintains a contingency table between feature values (x) and classes (y)
    over a window of size `window_size` and computes the Chi-squared statistic.

    Parameters
    ----------
    window_size
        Size of the rolling window.

    Examples
    --------
    >>> from river import stats

    >>> data = [
    ...     ("A", 0),
    ...     ("A", 0),
    ...     ("B", 1),
    ...     ("B", 1),
    ...     ("C", 0),
    ... ]

    >>> rchi = stats.RollingChiSquared(window_size=4)

    >>> for x, y in data[:4]:
    ...     rchi.update(x, y)

    >>> round(rchi.get(), 3)
    4.0

    >>> rchi.update(*data[4])
    >>> round(rchi.get(), 3)
    4.0

    """

    def __init__(self, window_size: int):
        utils.Rolling.__init__(self, obj=ChiSquared(), window_size=window_size)

    def update(self, x, y):
        super().update(x, y)

    def get(self):
        return super().get()

    @property
    def window_size(self):
        return self._window_size

    @property
    def p_value(self):
        return self.obj.p_value

    @property
    def degrees_of_freedom(self):
        return self.obj.degrees_of_freedom

    def is_significant(self, alpha=0.05):
        return self.obj.is_significant(alpha)
