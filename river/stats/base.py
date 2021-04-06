import abc

from river import base


class Statistic(base.Base):
    """A statistic."""

    # Define the format specification used for string representation.
    _fmt = ",.6f"  # Use commas to separate big numbers and show 6 decimals

    def get(self):
        """Return the current value of the statistic."""
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}: {self.get():{self._fmt}}".rstrip("0")


class Univariate(Statistic):
    """A univariate statistic measures a property of a variable."""

    def update(self, x):
        """Update and return the called instance."""
        raise NotImplementedError

    def revert(self, x):
        """Revert and return the called instance."""
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

    def update(self, x, y):
        """Update and return the called instance."""
        raise NotImplementedError
