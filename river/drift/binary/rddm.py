from __future__ import annotations

import math

from river import base

__all__ = ["RDDM"]


class RDDM(base.BinaryDriftAndWarningDetector):
    """Reactive Drift Detection Method (RDDM).

    RDDM is a drift detection method designed to improve the responsiveness
    of traditional methods like DDM by periodically shrinking the window
    to maintain a high standard deviation (reactivity).

    Parameters
    ----------
    n_min
        The minimum number of samples before detecting a drift.
    warning_thresh
        The threshold for signaling a warning (default 1.773).
    drift_thresh
        The threshold for signaling a drift (default 2.258).
    max_size_concept
        Maximum size of concept before forcing a silent window shrink.
    min_size_stable_concept
        Minimum size of stable concept (buffer size for recalculation).
    warn_limit
        Maximum instances allowed in warning zone before forcing a drift.

    Examples
    --------
    >>> from river import drift
    >>> rddm = drift.binary.RDDM(n_min=30)
    >>> stream = [0] * 100 + [1] * 100
    >>> drifts = []
    >>> for i, val in enumerate(stream):
    ...     rddm.update(val)
    ...     if rddm.drift_detected:
    ...         drifts.append(i)
    >>> len(drifts) >= 1
    True

    References
    ----------
    [^1]: Roberto S. M. de Barros, Silas G. T. C. Santos, and Paulo M. Gonçalves Jr.
        A Reactive Drift Detection Method. Expert Systems with Applications, 90, 344-355, 2017.

    """

    def __init__(
        self,
        n_min: int = 129,
        warning_thresh: float = 1.773,
        drift_thresh: float = 2.258,
        max_size_concept: int = 40000,
        min_size_stable_concept: int = 7000,
        warn_limit: int = 1400,
    ):
        super().__init__()
        self.n_min = n_min
        self.warning_thresh = warning_thresh
        self.drift_thresh = drift_thresh
        self.max_size_concept = max_size_concept
        self.min_size_stable_concept = min_size_stable_concept
        self.warn_limit = warn_limit
        self._reset()

    def _reset(self) -> None:
        super()._reset()
        self.stored_predictions = [0] * self.min_size_stable_concept
        self.num_stored_instances = 0
        self.first_pos = 0
        self.last_pos = -1
        self.last_warn_pos = -1
        self.last_warn_inst = -1
        self.inst_num = 0
        self.rddm_drift = False
        self._drift_detected = False
        self._warning_detected = False
        self._is_warning_edge = False

        self.m_p_min = float("inf")
        self.m_s_min = float("inf")
        self.m_ps_min = float("inf")
        self.reset_learning()

    def reset_learning(self) -> None:
        self.m_n = 1
        self.m_p = 1.0
        self.m_s = 0.0
        if self._drift_detected:
            self.m_p_min = float("inf")
            self.m_s_min = float("inf")
            self.m_ps_min = float("inf")

    def update(self, x: int | float) -> None:
        """Update the detector with a new binary error value (`1` for error, `0` for correct)."""
        if self.rddm_drift:
            self.reset_learning()
            if self.last_warn_pos != -1:
                self.first_pos = self.last_warn_pos
                self.num_stored_instances = self.last_pos - self.first_pos + 1
                if self.num_stored_instances <= 0:
                    self.num_stored_instances += self.min_size_stable_concept

            pos = self.first_pos
            for _ in range(self.num_stored_instances):
                pred = self.stored_predictions[pos]
                self.m_p = self.m_p + (pred - self.m_p) / self.m_n
                self.m_s = math.sqrt(self.m_p * (1.0 - self.m_p) / self.m_n)
                if (
                    self._drift_detected
                    and (self.m_n > self.n_min)
                    and (self.m_p + self.m_s < self.m_ps_min)
                ):
                    self.m_p_min = self.m_p
                    self.m_s_min = self.m_s
                    self.m_ps_min = self.m_p + self.m_s
                self.m_n += 1
                pos = (pos + 1) % self.min_size_stable_concept

            self.last_warn_pos = -1
            self.last_warn_inst = -1
            self.rddm_drift = False
            self._drift_detected = False

        self.last_pos = (self.last_pos + 1) % self.min_size_stable_concept
        self.stored_predictions[self.last_pos] = int(x)
        if self.num_stored_instances < self.min_size_stable_concept:
            self.num_stored_instances += 1
        else:
            self.first_pos = (self.first_pos + 1) % self.min_size_stable_concept
            if self.last_warn_pos == self.last_pos:
                self.last_warn_pos = -1

        self.m_p = self.m_p + (int(x) - self.m_p) / self.m_n
        self.m_s = math.sqrt(self.m_p * (1.0 - self.m_p) / self.m_n)

        self.inst_num += 1
        self.m_n += 1
        self._warning_detected = False
        self._is_warning_edge = False

        if self.m_n <= self.n_min:
            return

        if self.m_p + self.m_s < self.m_ps_min:
            self.m_p_min = self.m_p
            self.m_s_min = self.m_s
            self.m_ps_min = self.m_p + self.m_s

        if self.m_p + self.m_s > self.m_p_min + self.drift_thresh * self.m_s_min:
            self._drift_detected = True
            self.rddm_drift = True
            if self.last_warn_inst == -1:
                self.first_pos = self.last_pos
                self.num_stored_instances = 1
            return

        if self.m_p + self.m_s > self.m_p_min + self.warning_thresh * self.m_s_min:
            if (self.last_warn_inst != -1) and (
                self.last_warn_inst + self.warn_limit <= self.inst_num
            ):
                self._drift_detected = True
                self.rddm_drift = True
                self.first_pos = self.last_pos
                self.num_stored_instances = 1
                self.last_warn_pos = -1
                self.last_warn_inst = -1
                return

            self._warning_detected = True
            if self.last_warn_inst == -1:
                self._is_warning_edge = True
                self.last_warn_inst = self.inst_num
                self.last_warn_pos = self.last_pos
        else:
            self.last_warn_inst = -1
            self.last_warn_pos = -1

        if self.m_n > self.max_size_concept and not self._warning_detected:
            self.rddm_drift = True

    @property
    def is_warning_edge(self) -> bool:
        """Whether a new warning was detected at the current step (onset edge)."""
        return self._is_warning_edge
