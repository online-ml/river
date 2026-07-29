from __future__ import annotations

from river import stats
from river._river_rust import stats as _rust_stats


class PeakToPeak(stats.base.Univariate):
    """Running peak to peak (max - min).

    Examples
    --------

    >>> from river import stats

    >>> X = [1, -4, 3, -2, 2, 4]
    >>> ptp = stats.PeakToPeak()
    >>> for x in X:
    ...     ptp.update(x)
    ...     print(ptp.get())
    0.
    5.
    7.
    7.
    7.
    8.

    """

    def __init__(self) -> None:
        self._ptp: _rust_stats.RsPeakToPeak = _rust_stats.RsPeakToPeak()
        self._is_updated: bool = False

    @property
    def name(self) -> str:
        return "ptp"

    def update(self, x: float) -> None:
        self._ptp.update(x)
        if not self._is_updated:
            self._is_updated = True

    def get(self) -> float:
        if not self._is_updated:
            return 0.0
        return self._ptp.get()


class RollingPeakToPeak(stats.base.RollingUnivariate):
    """Running peak to peak (max - min) over a window.

    Parameters
    ----------
    window_size
        Size of the rolling window.

    Attributes
    ----------
    max : stats.RollingMax
        The running rolling max.
    min : stats.RollingMin
        The running rolling min.

    Examples
    --------

    >>> from river import stats

    >>> X = [1, -4, 3, -2, 2, 1]
    >>> ptp = stats.RollingPeakToPeak(window_size=2)
    >>> for x in X:
    ...     ptp.update(x)
    ...     print(ptp.get())
    0
    5
    7
    5
    4
    1

    """

    def __init__(self, window_size: int) -> None:
        self.max: stats.RollingMax = stats.RollingMax(window_size)
        self.min: stats.RollingMin = stats.RollingMin(window_size)

    @property
    def window_size(self) -> int:
        return self.max.window_size

    def update(self, x: float) -> None:
        self.max.update(x)
        self.min.update(x)

    def get(self) -> float | None:
        maximum = self.max.get()
        if maximum is None:
            return None
        minimum = self.min.get()
        if minimum is None:
            return None
        return maximum - minimum
