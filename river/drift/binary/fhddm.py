from __future__ import annotations

import math

from river import base


class FHDDM(base.BinaryDriftAndWarningDetector):
    """Fast Hoeffding Drift Detection Method.

    FHDDM is a drift detection method based on the Hoeffding's inequality which uses
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
    >>> fhddm = drift.binary.FHDDM()

    >>> # Simulate a data stream where the first 250 instances come from a uniform distribution
    >>> # of 1's and 0's
    >>> data_stream = rng.choices([0, 1], k=250)
    >>> # Increase the probability of 1's appearing in the next 250 instances
    >>> data_stream = data_stream + rng.choices([0, 1], k=250, weights=[0.9, 0.1])
    >>> # Update drift detector and verify if change is detected
    >>> for i, x in enumerate(data_stream):
    ...     fhddm.update(x)
    ...     if fhddm.drift_detected:
    ...         print(f"Change detected at index {i}")
    ...         break
    Change detected at index 315

    References
    ----------
    [^1]: A. Pesaranghader, H.L. Viktor, Fast Hoeffding Drift Detection Method for Evolving Data Streams. In the Proceedings of ECML-PKDD 2016.

    """
    def __init__(
        self, sliding_window_size: int = 100, confidence_level: float = 0.000001
    ):

        super().__init__()
        self.sliding_window_size = sliding_window_size
        self.confidence_level = confidence_level
        self._reset()

    def _reset(self):
        self._sliding_window = [0] * self.sliding_window_size
        self._pointer = 0
        self._epsilon = math.sqrt(
            (math.log(1 / self.confidence_level)) / (2 * self.sliding_window_size)
        )
        self._u_max = 0
        self._n_one = 0

    def update(self, x):
        if self.drift_detected:
            self._reset()

        if self._pointer < self.sliding_window_size:
            self._sliding_window[self._pointer] = x
            self._n_one += self._sliding_window[self._pointer]
            self._pointer += 1
        else:
            self._n_one -= self._sliding_window.pop(0)
            self._sliding_window.append(x)
            self._n_one += x

        if self._pointer == self.sliding_window_size:
            u = self._n_one / self.sliding_window_size
            self._u_max = u if (self._u_max < u) else self._u_max
            self._drift_detected = True if (self._u_max - u > self._epsilon) else False