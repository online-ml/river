from __future__ import annotations

import typing

from river import stats
from river.base import DriftDetector


class EWMA(DriftDetector):
    """Exponentially weighted moving average (EWMA) drift detector.

    An EWMA control chart for streaming data. The chart statistic
    Z = alpha * x + (1 - alpha) * Z tracks a fading mean of the incoming
    values, and an alarm is raised when Z moves away from its adaptive
    reference value by more than a multiple of the EWMA standard error.
    The running mean and variance are maintained with river.stats primitives,
    and the limits are corrected for the warm-up phase with the classic
    steady-state factor so that early samples do not over-fire.

    Parameters
    ----------
    alpha
        Fading factor in (0, 1]: the weight of the newest observation in the
        chart statistic. Smaller values give more influence to the stream's
        history and make the chart smoother.
    threshold
        Multiplier applied to the EWMA standard error to form the drift
        bound. Larger values make the detector more conservative.
    min_instances
        Number of warm-up samples to consume before drift testing begins.
    mode
        Direction to monitor: "up", "down", or "both" (default), mirroring
        the interface of PageHinkley.

    Examples
    --------
    >>> import random

    >>> from river import drift

    >>> detector = drift.EWMA(alpha=0.1, threshold=4, min_instances=50)

    >>> rng = random.Random(12345)

    >>> for i in range(2000):
    ...     value = rng.gauss(0, 1) if i < 1000 else rng.gauss(3, 1)
    ...     detector.update(value)
    ...     if detector.drift_detected:
    ...         print(f"Change detected at index {i}")
    Change detected at index 1010

    """

    _MODE_UP = "up"
    _MODE_DOWN = "down"
    _MODE_BOTH = "both"
    _VALID_MODES = [_MODE_UP, _MODE_DOWN, _MODE_BOTH]

    def __init__(
        self,
        alpha: float = 0.1,
        threshold: float = 4.0,
        min_instances: int = 30,
        mode: str = "both",
    ):
        super().__init__()

        if not 0 < alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")
        if threshold <= 0:
            raise ValueError("threshold must be greater than 0")
        if min_instances < 1:
            raise ValueError("min_instances must be at least 1")
        if mode not in self._VALID_MODES:
            raise ValueError(f"Invalid 'mode'. Valid values are: {self._VALID_MODES}")

        self.alpha = alpha
        self.threshold = threshold
        self.min_instances = min_instances
        self.mode = mode

        self._reset()

    def _reset(self) -> None:
        super()._reset()
        self._n = 0
        self._z = 0.0
        self._mean = stats.Mean()
        self._ewvar = stats.EWVar(self.alpha)

        if self.mode == self._MODE_UP:
            self._test_drift = self._test_increase
        elif self.mode == self._MODE_DOWN:
            self._test_drift = self._test_decrease
        else:
            self._test_drift = self._test_both

    def _test_increase(self, dev_up: float, dev_down: float, bound: float) -> bool:
        return dev_up > bound

    def _test_decrease(self, dev_up: float, dev_down: float, bound: float) -> bool:
        return dev_down > bound

    def _test_both(self, dev_up: float, dev_down: float, bound: float) -> bool:
        return dev_up > bound or dev_down > bound

    def update(self, x: int | float) -> None:
        if self.drift_detected:
            self._reset()

        self._n += 1

        if self._n == 1:
            self._z = x
        else:
            self._z = self.alpha * x + (1 - self.alpha) * self._z

        self._mean.update(x)
        self._ewvar.update(x)

        if self._n >= self.min_instances:
            variance = max(self._ewvar.get(), 0.0)
            sigma = variance**0.5
            factor = (self.alpha / (2 - self.alpha)) * (1 - (1 - self.alpha) ** (2 * self._n))
            bound = self.threshold * sigma * factor**0.5
            dev_up = self._z - self._mean.get()
            dev_down = -dev_up
            self._drift_detected = self._test_drift(dev_up, dev_down, bound)
        else:
            self._drift_detected = False

    @classmethod
    def _unit_test_params(cls) -> typing.Generator[dict[str, float]]:
        yield {}
        yield {"alpha": 0.5, "threshold": 2.5, "min_instances": 10}
