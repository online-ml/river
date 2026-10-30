from __future__ import annotations

import collections
import math

from river import base


class WSTD(base.BinaryDriftAndWarningDetector):
    """Wilcoxon rank-sum drift detector for binary error streams [^1].

    Compares recent and older windows of errors (`True`) and correct
    predictions (`False`). Uses average ranks and a two-sided normal
    approximation without variance correction for ties or continuity
    correction. Drift overrides warnings. The next update resets both
    windows after drift.

    Parameters
    ----------
    recent_window_size
        Recent window length, at least 2. Must fill before testing.
    older_window_size
        Older window capacity, at least 2.
    min_instances
        Older-window length needed for testing, from 2 to `older_window_size`.
    alpha_warning
        Warning p-value threshold.
    alpha_drift
        Drift p-value threshold. Must satisfy `0 < alpha_drift < alpha_warning < 1`.

    Examples
    --------
    >>> from river import drift
    >>> detector = drift.binary.WSTD()
    >>> for i, x in enumerate([0] * 200 + [1] * 60 + [0] * 200):
    ...     detector.update(x)
    ...     if detector.drift_detected:
    ...         print(i)
    210
    273

    References
    ----------
    [^1]: Barros et al. (2018). Wilcoxon Rank Sum Test Drift Detector.
        DOI 10.1016/j.neucom.2017.10.051.

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
        """Add the outcome of one prediction to the detector.

        Parameters
        ----------
        x
            Whether the prediction was incorrect. Use `True` or 1 for an
            error, and `False` or 0 for a correct prediction.

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
