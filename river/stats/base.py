from __future__ import annotations

import abc
import numbers

from river import base

__all__ = ["Univariate", "Bivariate"]


class Statistic(abc.ABC, base.Base):
    """A statistic."""

    # Define the format specification used for string representation.
    _fmt = ",.6f"  # Use commas to separate big numbers and show 6 decimals

    @abc.abstractmethod
    def get(self) -> float | None:
        """Return the current value of the statistic."""

    def __repr__(self):
        try:
            value = self.get()
        except NotImplementedError:
            value = None
        fmt_value = None if value is None else f"{value:{self._fmt}}".rstrip("0")
        return f"{self.__class__.__name__}: {fmt_value}"

    def __str__(self):
        return repr(self)

    def __gt__(self, other):
        return self.get() > other.get()


class Univariate(Statistic):
    """A univariate statistic measures a property of a variable."""

    @abc.abstractmethod
    def update(self, x: numbers.Number):
        """Update and return the called instance."""
        raise NotImplementedError

    @property
    def name(self):
        return self.__class__.__name__.lower()

    def __or__(self, other):
        from .link import Link

        return Link(left=self, right=other)


class RollingUnivariate(Univariate):
    """A rolling univariate statistic measures a property of a variable over a window."""

    @property
    @abc.abstractmethod
    def window_size(self):
        pass

    @property
    def name(self):
        return f"{super().name}_{self.window_size}"


class Bivariate(Statistic):
    """A bivariate statistic measures a relationship between two variables."""

    @abc.abstractmethod
    def update(self, x, y):
        """Update and return the called instance."""
