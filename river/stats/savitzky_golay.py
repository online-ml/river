from __future__ import annotations

import collections

from scipy.signal import savgol_coeffs

from river import stats


class SavitzkyGolay(stats.base.RollingUnivariate):
    """Savitzky-Golay smoothing filter.

    The Savitzky-Golay filter fits a polynomial of a given degree to a sliding window of
    points using least squares. The smoothed value of the most recent point is obtained by
    evaluating the fitted polynomial at that point. Because the fit is done at the last point
    of the window, the filter is causal and therefore compatible with online learning.

    This is a standard technique to smooth noisy signals while preserving their shape, for
    example when engineering features from sensor streams. It is a good fit whenever the
    underlying signal changes slowly relative to the sampling rate and the noise is
    higher-frequency than the trend you want to recover.

    Parameters
    ----------
    window_size
        Size of the sliding window. The polynomial is fitted to the last `window_size`
        observations.
    polyorder
        Degree of the polynomial to fit. Must be strictly smaller than `window_size`.

    Examples
    --------

    Let's smooth a noisy sine wave. The filtered output follows the underlying signal while
    removing most of the noise.

    >>> import math
    >>> from river import stats

    >>> stat = stats.SavitzkyGolay(window_size=5, polyorder=2)
    >>> for i, x in enumerate([math.sin(i / 5) + 0.2 * math.sin(i * 7) for i in range(20)]):
    ...     stat.update(x)
    ...     if i >= 4:
    ...         print(round(stat.get(), 4))
    0.7741
    0.7495
    0.7372
    0.7841
    0.8913
    1.0121
    1.0754
    1.0207
    0.8291
    0.5347
    0.2098
    -0.0673
    -0.2481
    -0.3338
    -0.3716
    -0.4273

    The filter returns `None` until the window is full.

    >>> stat = stats.SavitzkyGolay(window_size=3, polyorder=1)
    >>> stat.get() is None
    True

    Use case
    --------

    The filter is useful when you want to follow a slow-moving trend in a noisy signal.
    Consider a temperature probe that drifts upward over time but whose readings jump
    around on every sample. Smoothing the raw readings reveals the underlying trend, which
    can then be fed to a downstream model or threshold:

    >>> from river import stats

    >>> stat = stats.SavitzkyGolay(window_size=5, polyorder=2)
    >>> raw = [10.0, 12.3, 9.1, 14.2, 13.8, 16.1, 15.2, 18.9, 17.4, 20.1]
    >>> for x in raw:
    ...     stat.update(x)
    ...     if stat.get() is not None:
    ...         print(round(stat.get(), 3))
    14.194
    16.346
    15.171
    18.514
    17.851
    19.866

    The raw values swing between 9.1 and 20.1, whereas the smoothed values trace a steady
    upward drift. This makes it straightforward to spot when the probe crosses a threshold,
    for example.

    """

    def __init__(self, window_size: int = 5, polyorder: int = 2) -> None:
        self.window_size_value = window_size
        self.polyorder = polyorder
        self._coeffs = savgol_coeffs(
            window_length=window_size,
            polyorder=polyorder,
            pos=window_size - 1,
            use="dot",
        )
        self.window: collections.deque[float] = collections.deque(maxlen=window_size)

    @property
    def window_size(self) -> int:
        return self.window_size_value

    def update(self, x: float) -> None:
        self.window.append(x)

    def get(self) -> float | None:
        if len(self.window) < self.window_size:
            return None
        return float(sum(c * x for c, x in zip(self._coeffs, self.window)))
