from __future__ import annotations

import math

from river import base, stats


class DDM(base.BinaryDriftAndWarningDetector):
    """Drift Detection Method.

    DDM (Drift Detection Method) is a concept change detection method
    based on the PAC learning model premise, that the learner's error rate
    will decrease as the number of analysed samples increase, as long as the
    data distribution is stationary.

    If the algorithm detects an increase in the error rate, that surpasses
    a calculated threshold, either change is detected or the algorithm will
    warn the user that change may occur in the near future, which is called
    the warning zone.

    The detection threshold is calculated in function of two statistics,
    obtained when $(p_i + s_i)$ is minimum:

    * $p_{min}$: The minimum recorded error rate.

    * $s_{min}$: The minimum recorded standard deviation.

    At instant $i$, the detection algorithm uses:

    * $p_i$: The error rate at instant $i$.

    * $s_i$: The standard deviation at instant $i$.

    The conditions for entering the warning zone and detecting change are
    as follows [see implementation note below]:

    * if $p_i + s_i \\geq p_{min} + w_l * s_{min}$ -> Warning zone

    * if $p_i + s_i \\geq p_{min} + d_l * s_{min}$ -> Change detected

    In the above expressions, $w_l$ and $d_l$ represent, respectively, the warning and drift thresholds.

    **Input:** `x` is an entry in a stream of bits, where 1 indicates error/failure and 0
    represents correct/normal values.

    For example, if a classifier's prediction $y'$ is right or wrong w.r.t. the
    true target label $y$:

    - 0: Correct, $y=y'$

    - 1: Error, $y \\neq y'$

    Parameters
    ----------
    warm_start
        The minimum required number of analyzed samples so change can be detected. Warm start parameter
        for the drift detector.
    warning_threshold
        Threshold to decide if the detector is in a warning zone. The default value gives 95\\% of
        confidence level to the warning assessment.
    drift_threshold
        Threshold to decide if a drift was detected. The default value gives a 99\\% of confidence
        level to the drift assessment.

    Examples
    --------
    >>> import random
    >>> from river import drift

    >>> rng = random.Random(42)
    >>> ddm = drift.binary.DDM()

    >>> # Simulate a data stream where the first 1000 instances come from a uniform distribution
    >>> # of 1's and 0's
    >>> data_stream = rng.choices([0, 1], k=1000)
    >>> # Increase the probability of 1's appearing in the next 1000 instances
    >>> data_stream = data_stream + rng.choices([0, 1], k=1000, weights=[0.3, 0.7])

    >>> print_warning = True
    >>> # Update drift detector and verify if change is detected
    >>> for i, x in enumerate(data_stream):
    ...     _ = ddm.update(x)
    ...     if ddm.warning_detected and print_warning:
    ...         print(f"Warning detected at index {i}")
    ...         print_warning = False
    ...     if ddm.drift_detected:
    ...         print(f"Change detected at index {i}")
    ...         print_warning = True
    Warning detected at index 1084
    Change detected at index 1334
    Warning detected at index 1492

    References
    ----------
    [^1]: João Gama, Pedro Medas, Gladys Castillo, Pedro Pereira Rodrigues: Learning with Drift Detection. SBIA 2004: 286-295

    """

    def __init__(
        self, warm_start: int = 30, warning_threshold: float = 2.0, drift_threshold: float = 3.0
    ):
        super().__init__()
        self.warm_start = warm_start
        self.warning_threshold = warning_threshold
        self.drift_threshold = drift_threshold

        self._reset()

    def _reset(self):
        super()._reset()

        # Probability of error/failure
        self._p = stats.Mean()

        # Minimum values observed
        self._p_min = None
        self._s_min = None

        # The sum of p_min and s_min, to avoid calculating it every time
        self._ps_min = float("inf")

    def update(self, x):
        if self.drift_detected:
            self._reset()

        # Probability of error/failure
        self._p.update(x)

        p_i = self._p.get()
        n = self._p.n
        # Standard deviation of the error/failure: calculated using Bernoulli's properties
        s_i = math.sqrt(p_i * (1 - p_i) / n)

        if n > self.warm_start:
            if p_i + s_i <= self._ps_min:
                self._p_min = p_i
                self._s_min = s_i
                self._ps_min = self._p_min + self._s_min

            if p_i + s_i > self._p_min + self.warning_threshold * self._s_min:
                self._warning_detected = True
            else:
                self._warning_detected = False

            if p_i + s_i > self._p_min + self.drift_threshold * self._s_min:
                self._drift_detected = True
                self._warning_detected = False

        return self
