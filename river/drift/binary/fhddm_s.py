from __future__ import annotations

import math

from river import base, stats
from river.base.drift_detector import BinaryDriftDetector


class FHDDMS(base.BinaryDriftAndWarningDetector):
    """Stacking Fast Hoeffding Drift Detection Method.

    FHDDMS is a drift detection method based on the Hoeffding's inequality which uses
    the input average as estimator.

    **Input:** `x` is an entry in a stream of bits, where 1 indicates error/failure and 0
    represents correct/normal values.

    For example, if a classifier's prediction $y'$ is right or wrong w.r.t. the
    true target label $y$:

    - 0: Correct, $y=y'$

    - 1: Error, $y \\neq y'$

    *Implementation based on MOA.*
    Parameters
    ----------
    sliding_window_size
        The minimum required number of analyzed samples so change can be detected.
    confidence_level
        Threshold to decide if a drift was detected. The default value gives a 99\\% of confidence
        level to the drift assessment.

    Examples
    --------
    >>> import random
    >>> from river import drift

    >>> rng = random.Random(42)
    >>> fhddms = drift.binary.FHDDMS()

    >>> # Simulate a data stream where the first 1000 instances come from a uniform distribution
    >>> # of 1's and 0's
    >>> data_stream = rng.choices([0, 1], k=1000)
    >>> # Increase the probability of 1's appearing in the next 1000 instances
    >>> data_stream = data_stream + rng.choices([0, 1], k=1000, weights=[0.7, 0.3])
    >>> # Update drift detector and verify if change is detected
    >>> for i, x in enumerate(data_stream):
    ...     fhddms.update(x)
    ...     if fhddms.drift_detected:
    ...         print(f"Change detected at index {i}")
    ...         exit(0)
    Change detected at index 1057

    References
    ----------
    [^1]: Reservoir of Diverse Adaptive Learners and Stacking Fast Hoeffding Drift Detection Methods for Evolving Data Streams.

    """

    def __init__(
        self,
        stack_size: int = 4,
        short_window_size: int = 25,
        confidence_level: float = 0.000001,
    ):
        super().__init__()
        self.stack_size = stack_size
        self.short_window_size = short_window_size
        self.long_window_size = short_window_size * stack_size
        self.confidence_level = confidence_level

        self.stack = [0] * (self.short_window_size * self.stack_size)
        self.epsilon_s = math.sqrt(
            (math.log(1 / self.confidence_level)) / (2 * self.short_window_size)
        )
        self.epsilon_l = math.sqrt(
            (math.log(1 / self.confidence_level)) / (2 * self.long_window_size)
        )
        self.pointer = 0
        self.u_s_max = 0
        self.u_l_max = 0

    def _reset(self):
        self.stack = [0] * (self.short_window_size * self.stack_size)
        self.epsilon_s = math.sqrt(
            (math.log(1 / self.confidence_level)) / (2 * self.short_window_size)
        )
        self.epsilon_l = math.sqrt(
            (math.log(1 / self.confidence_level)) / (2 * self.long_window_size)
        )
        self.pointer = 0
        self.u_s_max = 0
        self.u_l_max = 0

    def update(self, x):
        if self.drift_detected:
            self._reset()

        if self.pointer < self.stack_size:
            self.stack[self.pointer] = x
            self.pointer += 1
        else:
            _ = self.stack.pop(0)
            self.stack.append(x)

        if self.pointer == self.stack_size:
            u_s = (
                sum(self.stack[self.stack_size - self.short_window_size :])
                / self.short_window_size
            )
            self.u_s_max = u_s if self.u_s_max < u_s else self.u_s_max
            u_l = sum(self.stack) / self.long_window_size
            self.u_l_max = u_l if self.u_l_max < u_l else self.u_l_max

            short_win_drift_status = (
                True if self.u_s_max - u_s > self.epsilon_s else False
            )
            long_win_drift_status = (
                True if self.u_l_max - u_l > self.epsilon_l else False
            )

            self._drift_detected = short_win_drift_status or long_win_drift_status
