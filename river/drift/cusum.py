from __future__ import annotations

import typing

from river import stats
from river.base import DriftDetector


class CUSUM(DriftDetector):
    r"""CUSUM (cumulative sum) control chart for change detection.

    Keeps two running sums of the deviations from the stream's mean, each offset by a slack
    term so that small fluctuations are absorbed and only sustained movement accumulates:

    .. math::

        S^+_i = \max(0,\; S^+_{i-1} + (x_i - \mu_i - k))

        S^-_i = \max(0,\; S^-_{i-1} - (x_i - \mu_i + k))

    A change is reported once either sum exceeds the decision interval :math:`h`. Because the
    sums reset at zero, a single outlier cannot trigger a detection: it takes a run of samples
    in the same direction. That makes CUSUM slower to react than a moving-average chart but far
    less prone to false alarms on noisy input.

    Both :math:`k` (slack) and :math:`h` (threshold) are expressed as multiples of the standard
    deviation of the warm-up window, so the detector is scale-free and needs no knowledge of the
    data's units. The classical values for standardized data are ``slack=0.5`` and ``threshold=5``.

    The reference mean and deviation are estimated over the first ``min_instances`` samples and
    then frozen: a running mean would chase the very shift being detected.

    At the classical settings the average run length between false alarms is a few hundred
    samples, so a long stable stream will occasionally alarm — that is inherent to CUSUM at
    ``threshold=5`` rather than a defect. Raising ``threshold`` trades detection delay for a
    longer false-alarm interval.

    This is the original tabular CUSUM. For a detector that tracks a drifting reference with a
    fading mean, see :class:`~river.drift.PageHinkley`; for a moving-average control chart, see
    :class:`~river.drift.EWMA`.

    Parameters
    ----------
    min_instances
        The minimum number of instances before detecting change.
    slack
        The slack :math:`k`, as a multiple of the reference standard deviation. Larger values
        make the detector less sensitive.
    threshold
        The decision interval :math:`h`, as a multiple of the reference standard deviation.
        Larger values delay detection and reduce false alarms.
    mode
        Whether to consider increases ("up"), decreases ("down") or both ("both").

    Examples
    --------
    >>> import random
    >>> from river import drift

    >>> rng = random.Random(12345)
    >>> cusum = drift.CUSUM()

    >>> # A stream that is stable, then shifts up by three standard deviations.
    >>> stable = [rng.gauss(0, 1) for _ in range(200)]
    >>> shifted = [rng.gauss(3, 1) for _ in range(200)]
    >>> data_stream = stable + shifted

    >>> for i, val in enumerate(data_stream):
    ...     cusum.update(val)
    ...     if cusum.drift_detected:
    ...         print(f"Change detected at index {i}")
    Change detected at index 201
    Change detected at index 251

    References
    ----------
    [^1]: E. S. Page. 1954. Continuous Inspection Schemes. Biometrika 41, 1/2 (1954), 100-115.

    """

    _MODE_UP = "up"
    _MODE_DOWN = "down"
    _MODE_BOTH = "both"
    _VALID_MODES = [_MODE_UP, _MODE_DOWN, _MODE_BOTH]

    def __init__(
        self,
        min_instances: int = 30,
        slack: float = 0.5,
        threshold: float = 5.0,
        mode: str = "both",
    ) -> None:
        super().__init__()

        if slack < 0:
            raise ValueError("slack must be greater than or equal to 0")
        if threshold <= 0:
            raise ValueError("threshold must be greater than 0")
        if min_instances < 1:
            raise ValueError("min_instances must be at least 1")
        if mode not in self._VALID_MODES:
            raise ValueError(f"Invalid 'mode'. Valid values are: {self._VALID_MODES}")

        self.min_instances = min_instances
        self.slack = slack
        self.threshold = threshold
        self.mode = mode

        self._reset()

    def _reset(self) -> None:
        super()._reset()
        self._n = 0
        self._mean = stats.Mean()
        self._var = stats.Var()
        self._sum_increase = 0.0
        self._sum_decrease = 0.0
        self._reference: float | None = None
        self._sigma = 0.0

        if self.mode == self._MODE_UP:
            self._test_drift = self._test_increase
        elif self.mode == self._MODE_DOWN:
            self._test_drift = self._test_decrease
        else:
            self._test_drift = self._test_both

    def _test_increase(self, increase: float, decrease: float, limit: float) -> bool:
        return increase > limit

    def _test_decrease(self, increase: float, decrease: float, limit: float) -> bool:
        return decrease > limit

    def _test_both(self, increase: float, decrease: float, limit: float) -> bool:
        return increase > limit or decrease > limit

    def update(self, x: int | float) -> None:
        if self.drift_detected:
            self._reset()

        self._n += 1
        self._mean.update(x)
        self._var.update(x)

        if self._n < self.min_instances:
            self._drift_detected = False
            return

        # The reference is estimated once over the warm-up window and then frozen. A running
        # mean would chase the very shift we are trying to detect, and would make the slack
        # and threshold parameters depend on the sample rather than on the process.
        if self._reference is None:
            self._reference = self._mean.get()
            self._sigma = max(self._var.get(), 0.0) ** 0.5

        k = self.slack * self._sigma
        limit = self.threshold * self._sigma
        deviation = x - self._reference

        self._sum_increase = max(0.0, self._sum_increase + deviation - k)
        self._sum_decrease = max(0.0, self._sum_decrease - deviation - k)

        self._drift_detected = self._test_drift(self._sum_increase, self._sum_decrease, limit)

    @classmethod
    def _unit_test_params(cls) -> typing.Generator[dict[str, float | int | str]]:
        yield {}
        yield {"min_instances": 10, "slack": 1.0, "threshold": 3.0}
        yield {"mode": "up"}
        yield {"mode": "down"}
