from __future__ import annotations

from river import stats
from river.base import DriftDetector


class PageHinkley(DriftDetector):
    """Page-Hinkley method for concept drift detection.

    This change detection method works by computing the observed values and their mean
    up to the current moment. Page-Hinkley does not signal warning zones, only change detections.

    This detector implements the CUSUM control chart for detecting changes. This implementation also supports
    the two-sided Page-Hinkley test to detect increasing and decreasing changes in the mean of the input
    values.

    Parameters
    ----------
    min_instances
        The minimum number of instances before detecting change.
    delta
        The delta factor for the Page-Hinkley test.
    threshold
        The change detection threshold (lambda).
    alpha
        The forgetting factor, used to weight the observed value and the mean.
    mode
        Whether to consider increases ("up"), decreases ("down") or both ("both") when monitoring the fading mean.

    Examples
    --------
    >>> import random
    >>> from river import drift

    >>> rng = random.Random(12345)
    >>> ph = drift.PageHinkley()

    >>> # Simulate a data stream composed by two data distributions
    >>> data_stream = rng.choices([0, 1], k=1000) + rng.choices(range(4, 8), k=1000)

    >>> # Update drift detector and verify if change is detected
    >>> for i, val in enumerate(data_stream):
    ...     _ = ph.update(val)
    ...     if ph.drift_detected:
    ...         print(f"Change detected at index {i}, input value: {val}")
    Change detected at index 1006, input value: 5

    References
    ----------
    [^1]: E. S. Page. 1954. Continuous Inspection Schemes. Biometrika 41, 1/2 (1954), 100-115.
    [^2]: Sebastião, R., & Fernandes, J. M. (2017, June). Supporting the Page-Hinkley test with
    empirical mode decomposition for change detection. In International Symposium on Methodologies
    for Intelligent Systems (pp. 492-498). Springer, Cham.

    """

    _MODE_UP = "up"
    _MODE_DOWN = "down"
    _MODE_BOTH = "both"
    _VALID_MODES = [_MODE_UP, _MODE_DOWN, _MODE_BOTH]

    def __init__(
        self,
        min_instances: int = 30,
        delta: float = 0.005,
        threshold: float = 50.0,
        alpha: float = 1 - 0.0001,
        mode: str = "both",
    ):
        super().__init__()
        self.min_instances = min_instances
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha

        if mode not in self._VALID_MODES:
            raise ValueError(f"Invalid 'mode'. Valid values are: {self._VALID_MODES}")

        self.mode = mode

        self._reset()

    def _reset(self):
        super()._reset()
        self._x_mean = stats.Mean()
        self._sum_increase = 0.0
        self._sum_decrease = 0.0

        self._min_increase = float("inf")
        self._max_decrease = -1

        if self.mode == self._MODE_UP:
            self._test_drift = self._test_increase
        elif self.mode == self._MODE_DOWN:
            self._test_drift = self._test_decrease
        else:
            self._test_drift = self._test_both

    def _test_increase(self, test_increase, test_decrease) -> bool:
        return test_increase > self.threshold

    def _test_decrease(self, test_increase, test_decrease) -> bool:
        return test_decrease > self.threshold

    def _test_both(self, test_increase, test_decrease) -> bool:
        return test_increase > self.threshold or test_decrease > self.threshold

    def update(self, x):
        if self.drift_detected:
            self._reset()

        self._x_mean.update(x)
        dev = x - self._x_mean.get()

        self._sum_increase = self.alpha * self._sum_increase + dev - self.delta
        self._sum_decrease = self.alpha * self._sum_decrease + dev + self.delta

        if self._sum_increase < self._min_increase:
            self._min_increase = self._sum_increase

        if self._sum_decrease > self._max_decrease:
            self._max_decrease = self._sum_decrease

        if self._x_mean.n >= self.min_instances:
            test_increase = self._sum_increase - self._min_increase
            test_decrease = self._max_decrease - self._sum_decrease
            self._drift_detected = self._test_drift(test_increase, test_decrease)

        return self
