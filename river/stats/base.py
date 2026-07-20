from __future__ import annotations

import abc
from typing import Generic, TypeVar

from river import base

__all__ = ["Bivariate", "Link", "RollingUnivariate", "Statistic", "Univariate"]

# The type of the values a statistic produces. Defaults to float so that existing
# annotations such as `stats.base.Statistic` keep meaning "a float statistic".
R = TypeVar("R", default=float)
# The type of the values a statistic observes.
T = TypeVar("T", default=float)
# A third type to represent consumed/produced values.
S = TypeVar("S", default=float)


class Statistic(abc.ABC, base.Base, Generic[R]):
    """A statistic."""

    # Define the format specification used for string representation.
    _fmt = ",.6f"  # Use commas to separate big numbers and show 6 decimals

    @abc.abstractmethod
    def get(self) -> R:
        """Return the current value of the statistic."""

    def __repr__(self) -> str:
        try:
            value = self.get()
        except NotImplementedError:
            value = None
        fmt_value = None if value is None else f"{value:{self._fmt}}".rstrip("0")
        return f"{self.__class__.__name__}: {fmt_value}"

    def __str__(self) -> str:
        return repr(self)

    def __gt__(self: Statistic[float], other: Statistic[float]) -> bool:
        return self.get() > other.get()


class Univariate(Statistic[R], Generic[T, R]):
    """A univariate statistic measures a property of a variable."""

    @abc.abstractmethod
    def update(self, x: T) -> None:
        """Update the called instance."""

    @property
    def name(self) -> str:
        return self.__class__.__name__.lower()

    def __or__(self, other: Univariate[R, S]) -> Univariate[T, S]:
        return Link(left=self, right=other)


class Link(Univariate[T, S], Generic[T, R, S]):
    """A link joins two univariate statistics as a sequence.

    This can be used to pipe the output of one statistic to the input of another. This can be used,
    for instance, to calculate the mean of the variance of a variable. It can also be used to
    compute shifted statistics by piping statistics with an instance of `stats.Shift`.

    Note that a link is not meant to be instantiated via this class definition. Instead, users can
    link statistics together via the `|` operator.

    Parameters
    ----------
    left
    right
        The output from `left`'s `get` method is passed to `right`'s `update` method if `left`'s
        `get` method doesn't produce `None.`

    Examples
    --------

    >>> from river import stats
    >>> stat = stats.Shift(1) | stats.Mean()

    No values have been seen, therefore `get` defaults to the initial value of `stats.Mean`,
    which is 0.

    >>> stat.get()
    0.

    Let us now call `update`.

    >>> stat.update(1)

    The output from `get` will still be 0. The reason is that `stats.Shift` has not enough
    values, and therefore outputs its default value, which is `None`. The `stats.Mean`
    instance is therefore not updated.

    >>> stat.get()
    0.0

    On the next call to `update`, the `stats.Shift` instance has seen enough values, and
    therefore the mean can be updated. The mean is therefore equal to 1, because that's the
    only value from the past.

    >>> stat.update(3)
    >>> stat.get()
    1.0

    On the subsequent call to update, the mean will be updated with the value 3.

    >>> stat.update(4)
    >>> stat.get()
    2.0

    Note that composing statistics returns a new statistic with its own name.

    >>> stat.name
    'mean_of_shift_1'

    """

    def __init__(self, left: Univariate[T, R], right: Univariate[R, S]) -> None:
        self.left = left
        self.right = right

    def update(self, x: T) -> None:
        self.left.update(x)
        y = self.left.get()
        self.right.update(y)

    def get(self) -> S:
        return self.right.get()

    @property
    def name(self) -> str:
        return f"{self.right.name}_of_{self.left.name}"

    def __repr__(self) -> str:
        return repr(self.right)


class RollingUnivariate(Univariate[T, R]):
    """A rolling univariate statistic measures a property of a variable over a window."""

    @property
    @abc.abstractmethod
    def window_size(self) -> int:
        """The size of the rolling window."""

    @property
    def name(self) -> str:
        return f"{super().name}_{self.window_size}"


class Bivariate(Statistic[R], Generic[T, S, R]):
    """A bivariate statistic measures a relationship between two variables."""

    @abc.abstractmethod
    def update(self, x: T, y: S) -> None:
        """Update the called instance."""
