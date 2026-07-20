from __future__ import annotations

import math

from river import stats, utils


class Max(stats.base.Univariate):
    """Running max.

    Attributes
    ----------
    max : float
        The current max.

    Examples
    --------

    >>> from river import stats

    >>> X = [1, -4, 3, -2, 5, -6]
    >>> maximum = stats.Max()
    >>> for x in X:
    ...     maximum.update(x)
    ...     print(maximum.get())
    1
    1
    3
    3
    5
    5

    """

    def __init__(self) -> None:
        self.max: float = -math.inf

    def update(self, x: float) -> None:
        if x > self.max:
            self.max = x

    def get(self) -> float:
        return self.max


class RollingMax(stats.base.RollingUnivariate):
    """Running max over a window.

    Parameters
    ----------
    window_size
        Size of the rolling window.

    Examples
    --------

    >>> from river import stats

    >>> X = [1, -4, 3, -2, 2, 1]
    >>> rolling_max = stats.RollingMax(window_size=2)
    >>> for x in X:
    ...     rolling_max.update(x)
    ...     print(rolling_max.get())
    1
    1
    3
    3
    2
    2

    """

    def __init__(self, window_size: int) -> None:
        self.window: utils.SortedWindow[float] = utils.SortedWindow(size=window_size)

    @property
    def window_size(self) -> int:
        return self.window.size

    def update(self, x: float) -> None:
        self.window.append(x)

    def get(self) -> float:
        try:
            return self.window[-1]
        except IndexError:
            raise stats.NotEnoughSamples


class AbsMax(stats.base.Univariate):
    """Running absolute max.

    Attributes
    ----------
    abs_max : float
        The current absolute max.

    Examples
    --------

    >>> from river import stats

    >>> X = [1, -4, 3, -2, 5, -6]
    >>> abs_max = stats.AbsMax()
    >>> for x in X:
    ...     abs_max.update(x)
    ...     print(abs_max.get())
    1
    4
    4
    4
    5
    6

    """

    def __init__(self) -> None:
        self.abs_max: float = 0.0

    def update(self, x: float) -> None:
        if abs(x) > self.abs_max:
            self.abs_max = abs(x)

    def get(self) -> float:
        return self.abs_max


class RollingAbsMax(stats.base.RollingUnivariate):
    """Running absolute max over a window.

    Parameters
    ----------
    window_size
        Size of the rolling window.

    Examples
    --------

    >>> from river import stats

    >>> X = [1, -4, 3, -2, 2, 1]
    >>> rolling_absmax = stats.RollingAbsMax(window_size=2)
    >>> for x in X:
    ...     rolling_absmax.update(x)
    ...     print(rolling_absmax.get())
    1
    4
    4
    3
    2
    2

    """

    def __init__(self, window_size: int) -> None:
        self.window: utils.SortedWindow[float] = utils.SortedWindow(size=window_size)

    @property
    def window_size(self) -> int:
        return self.window.size

    def update(self, x: float) -> None:
        self.window.append(abs(x))

    def get(self) -> float:
        try:
            return self.window[-1]
        except IndexError:
            raise stats.NotEnoughSamples
