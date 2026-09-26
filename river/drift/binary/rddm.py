from __future__ import annotations

import math

from river import base


class RDDM(base.BinaryDriftAndWarningDetector):
    """Reactive Drift Detection Method.

    DDM latches onto the smallest error rate it has ever seen and only warns once the current
    rate climbs well past that global minimum. After a long, stable concept the reference becomes
    very small and the thresholds drift out of reach, so the error rate can degrade substantially
    before DDM reacts.

    RDDM keeps a buffer of recent error rates instead of trusting the global minimum. When drift
    is declared, the *next* call replays that buffer to rebuild the running statistics and
    re-derive the reference from the argmin of :math:`p + s` within the replayed window, which is
    what lets the detector re-anchor after a long stable stretch rather than staying deaf to it.

    Three things force a change:

    * the standard DDM out-of-control test fires;
    * a warning-zone streak stays open for :math:``warning_limit`` instances, and is then escalated;
    * the current concept exceeds :math:``max_concept_size`` instances without a warning, which
      declares the concept itself too long to trust.

    **Provenance.** The paper is paywalled, so this is a port of the authors' own reference
    implementation in MOA (`RDDM.java`), cross-checked against Frouros' independent
    implementation. The defaults are MOA's. The buffered, deferred-replay structure means this
    cannot be expressed as a small delta on :class:`~river.drift.binary.DDM`.

    Unlike river's other binary detectors, a detection does not clear the state: the buffer is the
    thing that makes the re-anchoring work, so the reset is deferred by one call, as in the
    reference implementation.

    **Input:** `x` is an entry in a stream of bits, where 1 indicates error/failure and 0
    represents correct/normal values.

    Parameters
    ----------
    warm_start
        Number of instances below which no test is applied.
    warning_threshold
        Multiplier of :math:`s_{min}` that opens the warning zone.
    drift_threshold
        Multiplier of :math:`s_{min}` that declares drift.
    buffer_size
        Number of recent error rates retained for the reactive replay.
    max_concept_size
        Concept length beyond which a drift is declared if the warning zone is not active.
    warning_limit
        Number of instances a warning-zone streak may stay open before it escalates to drift.

    Examples
    --------
    >>> import random
    >>> from river import drift

    >>> rng = random.Random(42)
    >>> rddm = drift.binary.RDDM()

    >>> # A long, stable, low-error concept, then a jump to a much higher error rate.
    >>> data_stream = [1 if rng.random() < 0.1 else 0 for _ in range(1000)]
    >>> data_stream += [1 if rng.random() < 0.5 else 0 for _ in range(1000)]

    >>> for i, x in enumerate(data_stream):
    ...     rddm.update(x)
    ...     if rddm.drift_detected:
    ...         print(f"Change detected at index {i}")
    Change detected at index 1021

    References
    ----------
    [^1]: R. S. M. Barros, L. M. Cabral, R. M. Gonçalves, J. L. Santos. 2017. RDDM: Reactive
        drift detection method. Expert Systems with Applications 90: 344-355.

    """

    def __init__(
        self,
        warm_start: int = 129,
        warning_threshold: float = 1.773,
        drift_threshold: float = 2.258,
        buffer_size: int = 7000,
        max_concept_size: int = 40000,
        warning_limit: int = 1400,
    ):
        super().__init__()

        if warm_start < 1:
            raise ValueError("warm_start must be at least 1")
        if warning_threshold <= 0:
            raise ValueError("warning_threshold must be greater than 0")
        if drift_threshold <= warning_threshold:
            raise ValueError("drift_threshold must be greater than warning_threshold")
        if buffer_size < 1:
            raise ValueError("buffer_size must be at least 1")
        if max_concept_size < 1:
            raise ValueError("max_concept_size must be at least 1")
        if warning_limit < 1:
            raise ValueError("warning_limit must be at least 1")

        self.warm_start = warm_start
        self.warning_threshold = warning_threshold
        self.drift_threshold = drift_threshold
        self.buffer_size = buffer_size
        self.max_concept_size = max_concept_size
        self.warning_limit = warning_limit

        self._reset()

    def _reset(self) -> None:
        super()._reset()

        # Running error statistics, maintained directly rather than via stats.Mean so that the
        # reactive replay can rewind and re-derive them.
        self._n = 0
        self._p = 0.0
        self._s = 0.0

        # The reference, always updated together so that _s_min belongs to the _p_min that
        # produced the minimum. Tracking them independently would give a different detector.
        self._p_min = float("inf")
        self._s_min = float("inf")
        self._ps_min = float("inf")

        # Ring buffer of recent error rates and the bookkeeping needed to replay it.
        self._buffer = bytearray(self.buffer_size)
        self._first_pos = 0
        self._last_pos = -1
        self._num_stored = 0

        # Warning streak: the instance and buffer position at which the zone was entered.
        self._last_warn_inst = -1
        self._last_warn_pos = -1

        # Set when drift is declared, cleared once the reactive replay has run.
        self._pending = False
        self._change_detected = False

        self._inst = 0

    def _react(self) -> None:
        """Replay the buffer to rebuild the statistics and re-anchor the reference."""
        if self._change_detected:
            self._p_min = self._s_min = self._ps_min = float("inf")

        self._n = 1
        self._p = 1.0
        self._s = 0.0

        if self._last_warn_pos != -1:
            # Restart the replay from where the warning zone was entered.
            self._first_pos = self._last_warn_pos
            self._num_stored = self._last_pos - self._first_pos + 1
            if self._num_stored <= 0:
                self._num_stored += self.buffer_size

        pos = self._first_pos
        for _ in range(self._num_stored):
            self._p += (self._buffer[pos] - self._p) / self._n
            self._s = math.sqrt(self._p * (1 - self._p) / self._n)
            if (
                self._change_detected
                and self._n > self.warm_start
                and self._p + self._s < self._ps_min
            ):
                self._p_min = self._p
                self._s_min = self._s
                self._ps_min = self._p + self._s
            self._n += 1
            pos = (pos + 1) % self.buffer_size

        self._last_warn_pos = -1
        self._last_warn_inst = -1
        self._pending = False
        self._change_detected = False

    def update(self, x: bool) -> None:
        if self._pending:
            self._react()

        self._inst += 1

        # Store the instance, growing the buffer until it is full and then sliding.
        self._last_pos = (self._last_pos + 1) % self.buffer_size
        self._buffer[self._last_pos] = int(x)
        if self._num_stored < self.buffer_size:
            self._num_stored += 1
        else:
            self._first_pos = (self._first_pos + 1) % self.buffer_size
            if self._last_warn_pos == self._last_pos:
                # The anchor was just overwritten, so the streak is older than the buffer.
                self._last_warn_pos = -1

        self._n += 1
        self._p += (float(x) - self._p) / self._n
        self._s = math.sqrt(self._p * (1 - self._p) / self._n)

        self._warning_detected = False
        self._drift_detected = False

        if self._n <= self.warm_start:
            return

        if self._p + self._s < self._ps_min:
            self._p_min = self._p
            self._s_min = self._s
            self._ps_min = self._p + self._s

        if self._p + self._s > self._p_min + self.drift_threshold * self._s_min:
            self._drift_detected = True
            self._pending = True
            self._change_detected = True
            if self._last_warn_pos == -1:
                self._first_pos = self._last_pos
                self._num_stored = 1
            return

        if self._p + self._s > self._p_min + self.warning_threshold * self._s_min:
            self._warning_detected = True
            if self._last_warn_inst == -1:
                self._last_warn_inst = self._inst
                self._last_warn_pos = self._last_pos
            if self._last_warn_inst + self.warning_limit <= self._inst:
                # The zone has been open too long; treat it as a change.
                self._drift_detected = True
                self._pending = True
                self._change_detected = True
                self._first_pos = self._last_pos
                self._num_stored = 1
                self._last_warn_pos = -1
                self._last_warn_inst = -1
                return
        else:
            self._last_warn_inst = -1
            self._last_warn_pos = -1

        if self._n > self.max_concept_size and not self._warning_detected:
            self._drift_detected = True
            self._pending = True
