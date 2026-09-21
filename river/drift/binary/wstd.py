from __future__ import annotations

import collections
import math

from river import base


class WSTD(base.BinaryDriftAndWarningDetector):
    r"""Wilcoxon Rank Sum Test drift detector.

    WSTD monitors a stream of boolean indicators and signals a warning or a drift when a
    Wilcoxon rank-sum test finds the distribution of the most recent observations to be
    significantly different from that of the older ones. It is a variant of STEPD in which
    the test of proportions is replaced by the Wilcoxon rank-sum test, whose normal
    approximation allows an exact and cheap evaluation on binary inputs.

    A sliding window of the last `recent_window_size` + `older_window_size` observations is
    split into a recent sub-window (the last `recent_window_size` values) and an older
    sub-window (up to `older_window_size` preceding values). Let $n_1$ and $n_2$ be the sizes
    of the smaller and the larger sub-window, $N = n_1 + n_2$, and $R$ the rank sum of the
    smaller sub-window in the pooled sample (ties receive mid-ranks). Under the null
    hypothesis that both sub-windows are drawn from the same distribution:

    $$
    z = \frac{R - \mu_R}{\sigma_R}, \quad \mu_R = \frac{n_1 (N + 1)}{2}, \quad
    \sigma_R = \sqrt{\frac{n_1 n_2 (N + 1)}{12}}
    $$

    Drift is signaled when the two-sided p-value of $z$ drops below `alpha_drift`, and a
    warning when it drops below `alpha_warning`. The windows are reset after a drift.

    **Input:** `x` is an entry in a stream of bits, where 1 indicates error/failure and 0
    represents correct/normal values.

    For example, if a classifier's prediction $y'$ is right or wrong w.r.t. the
    true target label $y$:

    - 0: Correct, $y=y'$

    - 1: Error, $y \\neq y'$

    Parameters
    ----------
    recent_window_size
        The number of most recent observations that form the recent sub-window.
    older_window_size
        The maximum number of observations preceding the recent sub-window that form the
        older sub-window.
    min_instances
        The minimum number of observations in the older sub-window required before the
        statistical test is applied.
    alpha_warning
        Significance level below which a warning is signaled. Must be greater than
        `alpha_drift`.
    alpha_drift
        Significance level below which a drift is signaled.

    Examples
    --------
    >>> import random
    >>> from river import drift

    >>> rng = random.Random(42)
    >>> wstd = drift.binary.WSTD()

    >>> # Simulate a data stream where the first 250 instances come from a uniform distribution
    >>> # of 1's and 0's
    >>> data_stream = rng.choices([0, 1], k=250)
    >>> # Increase the probability of 1's appearing in the next 250 instances
    >>> data_stream = data_stream + rng.choices([0, 1], k=250, weights=[0.1, 0.9])

    >>> print_warning = True
    >>> # Update drift detector and verify if change is detected
    >>> for i, x in enumerate(data_stream):
    ...     wstd.update(x)
    ...     if wstd.warning_detected and print_warning:
    ...         print(f"Warning detected at index {i}")
    ...         print_warning = False
    ...     if wstd.drift_detected:
    ...         print(f"Change detected at index {i}")
    ...         print_warning = True
    Warning detected at index 265
    Change detected at index 277

    References
    ----------
    [^1]: Ricardo S. M. de Barros, José I. Gómez Hidalgo, and Daniel R. L. Cabral. Wilcoxon Rank Sum Test Drift Detector. In Neurocomputing, volume 275, pages 1954-1963, 2018. doi:10.1016/j.neucom.2017.10.051.

    """

    def __init__(
        self,
        recent_window_size: int = 30,
        older_window_size: int = 120,
        min_instances: int = 30,
        alpha_warning: float = 0.05,
        alpha_drift: float = 0.003,
    ):
        super().__init__()

        if recent_window_size < 2:
            raise ValueError("recent_window_size must be at least 2.")

        if older_window_size < 2:
            raise ValueError("older_window_size must be at least 2.")

        if min_instances < 2:
            raise ValueError("min_instances must be at least 2.")

        if min_instances > older_window_size:
            raise ValueError("min_instances must not be greater than older_window_size.")

        if not 0 < alpha_drift < alpha_warning < 1:
            raise ValueError(
                "alpha_warning and alpha_drift must be in (0, 1) and alpha_drift must be "
                "strictly smaller than alpha_warning."
            )

        self.recent_window_size = recent_window_size
        self.older_window_size = older_window_size
        self.min_instances = min_instances
        self.alpha_warning = alpha_warning
        self.alpha_drift = alpha_drift

        self._reset()

    def _reset(self) -> None:
        super()._reset()
        self._recent: collections.deque[bool] = collections.deque(maxlen=self.recent_window_size)
        self._older: collections.deque[bool] = collections.deque(maxlen=self.older_window_size)
        self._ones_recent = 0
        self._ones_older = 0
        self.p_value = 1.0

    def update(self, x: bool) -> None:
        """Update the change detector with a single data point.

        Parameters
        ----------
        x
            This parameter indicates whether the last sample analyzed was correctly classified
            or not. 1 indicates an error (miss-classification).

        """
        if self.drift_detected:
            self._reset()

        self._drift_detected = False
        self._warning_detected = False

        x = bool(x)

        if len(self._recent) == self.recent_window_size:
            evicted = self._recent[0]
            if len(self._older) == self.older_window_size:
                self._ones_older -= self._older[0]
            self._older.append(evicted)
            self._ones_older += evicted
            self._ones_recent -= evicted

        self._recent.append(x)
        self._ones_recent += x

        n_recent = len(self._recent)
        n_older = len(self._older)
        if n_recent < self.recent_window_size or n_older < self.min_instances:
            return

        ones = self._ones_recent + self._ones_older
        zeros = n_recent + n_older - ones
        rank_zero = (zeros + 1) / 2
        rank_one = zeros + (ones + 1) / 2

        rank_sum_recent = (n_recent - self._ones_recent) * rank_zero + self._ones_recent * rank_one
        rank_sum_older = (n_older - self._ones_older) * rank_zero + self._ones_older * rank_one

        if n_recent < n_older:
            n1, rank_sum = n_recent, rank_sum_recent
        elif n_older < n_recent:
            n1, rank_sum = n_older, rank_sum_older
        else:
            n1, rank_sum = n_recent, min(rank_sum_recent, rank_sum_older)
        n2 = n_recent + n_older - n1

        mean = n1 * (n_recent + n_older + 1) / 2
        std = math.sqrt(n1 * n2 * (n_recent + n_older + 1) / 12)
        z = (rank_sum - mean) / std
        self.p_value = math.erfc(abs(z) / math.sqrt(2))

        if self.p_value < self.alpha_drift:
            self._drift_detected = True
        elif self.p_value < self.alpha_warning:
            self._warning_detected = True
