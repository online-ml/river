from __future__ import annotations

import typing

from river import stats
from river.base import DriftDetector


class Shewhart(DriftDetector):
    r"""Shewhart individuals control chart (Shewhart, 1939).

    The simplest of the classical control charts. A center line is drawn at the reference mean
    and fixed limits are placed :math:`k` standard deviations away; an observation falling
    outside the limits is flagged.

    Unlike :class:`~river.drift.EWMA` and :class:`~river.drift.CUSUM`, this chart accumulates no
    evidence: each sample is judged on its own. That makes it the fastest to react to a large
    abrupt shift and the noisiest on ordinary variation — a single outlier is enough, so the
    limits are conventionally set wide (the classical choice is :math:`k = 3`).

    The center line and deviation are estimated over the warm-up window and then frozen, which
    keeps the limits a property of the baseline rather than of the sample being tested. Use
    :class:`~river.drift.EWMA` for sustained shifts, or :class:`~river.drift.CUSUM` when a
    single outlier should not be enough.

    Parameters
    ----------
    k
        Width of the control limits, in multiples of the reference standard deviation. The
        classical Shewhart choice is 3.
    min_instances
        The minimum number of instances before detecting change.
    mode
        Whether to consider increases ("up"), decreases ("down") or both ("both").

    Examples
    --------
    >>> import random
    >>> from river import drift

    >>> rng = random.Random(12345)
    >>> chart = drift.Shewhart(k=4)

    >>> # A stream that is stable, then shifts up by five standard deviations.
    >>> stable = [rng.gauss(0, 1) for _ in range(300)]
    >>> shifted = [rng.gauss(5, 1) for _ in range(200)]
    >>> data_stream = stable + shifted

    >>> for i, val in enumerate(data_stream):
    ...     chart.update(val)
    ...     if chart.drift_detected:
    ...         print(f"Change detected at index {i}")
    Change detected at index 300

    References
    ----------
    [^1]: W. A. Shewhart. 1939. Statistical Method. In Statistical Quality Control.
        McGraw-Hill.

    """

    _MODE_UP = "up"
    _MODE_DOWN = "down"
    _MODE_BOTH = "both"
    _VALID_MODES = [_MODE_UP, _MODE_DOWN, _MODE_BOTH]

    def __init__(
        self,
        k: float = 3.0,
        min_instances: int = 30,
        mode: str = "both",
    ) -> None:
        super().__init__()

        if k <= 0:
            raise ValueError("k must be greater than 0")
        if min_instances < 1:
            raise ValueError("min_instances must be at least 1")
        if mode not in self._VALID_MODES:
            raise ValueError(f"Invalid 'mode'. Valid values are: {self._VALID_MODES}")

        self.k = k
        self.min_instances = min_instances
        self.mode = mode

        self._reset()

    def _reset(self) -> None:
        super()._reset()
        self._n = 0
        self._mean = stats.Mean()
        self._var = stats.Var()
        self._center = 0.0
        self._sigma = 0.0

        if self.mode == self._MODE_UP:
            self._test_drift = self._test_increase
        elif self.mode == self._MODE_DOWN:
            self._test_drift = self._test_decrease
        else:
            self._test_drift = self._test_both

    def _test_increase(self, deviation: float, limit: float) -> bool:
        return deviation > limit

    def _test_decrease(self, deviation: float, limit: float) -> bool:
        return -deviation > limit

    def _test_both(self, deviation: float, limit: float) -> bool:
        return deviation > limit or -deviation > limit

    def update(self, x: int | float) -> None:
        if self.drift_detected:
            self._reset()

        self._n += 1
        self._mean.update(x)
        self._var.update(x)

        if self._n < self.min_instances:
            self._drift_detected = False
            return

        if self._n == self.min_instances:
            # Freeze the center line and the limits on the last warm-up sample.
            self._center = self._mean.get()
            self._sigma = max(self._var.get(), 0.0) ** 0.5

        self._drift_detected = self._test_drift(x - self._center, self.k * self._sigma)

    @classmethod
    def _unit_test_params(cls) -> typing.Generator[dict[str, float | int | str]]:
        yield {}
        yield {"k": 2.0, "min_instances": 10}
        yield {"mode": "up"}
        yield {"mode": "down"}
