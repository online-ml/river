from __future__ import annotations

import collections
import itertools
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
        Confidence level used to determine the epsilon coefficient in Hoeffding’s inequality. The default value gives a 99\\% of confidence
        level to the drift assessment.
    short_window_size
        The size of the short window size that it is used in a Stacking version of FHDDM [^2].

    Examples
    --------
    >>> import random
    >>> from river import drift

    >>> rng = random.Random(42)
    >>> # Traditional FHDDM [1]
    >>> fhddm = drift.binary.FHDDM()
    >>> # Stacking FHDDM [2]
    >>> fhddm_s = drift.binary.FHDDM(short_window_size = 20)
    >>> # Simulate a data stream where the first 250 instances come from a uniform distribution
    >>> # of 1's and 0's
    >>> data_stream = rng.choices([0, 1], k=250)
    >>> # Increase the probability of 1's appearing in the next 250 instances
    >>> data_stream = data_stream + rng.choices([0, 1], k=250, weights=[0.9, 0.1])
    >>> # Update drift detector and verify if change is detected
    >>> for i, x in enumerate(data_stream):
    ...     fhddm.update(x)
    ...     fhddm_s.update(x)
    ...     if fhddm.drift_detected or fhddm_s.drift_detected:
    ...         print(f"Change detected at index {i}")
    Change detected at index 279
    Change detected at index 315

    References
    ----------
    [^1]: A. Pesaranghader, H.L. Viktor, Fast Hoeffding Drift Detection Method for Evolving Data Streams. In the Proceedings of ECML-PKDD 2016.
    [^2]: Reservoir of Diverse Adaptive Learners and Stacking Fast Hoeffding Drift Detection Methods for Evolving Data Streams.

    """

    def __init__(
        self,
        sliding_window_size: int = 100,
        confidence_level: float = 0.000001,
        short_window_size: int | None = None,
    ):
        super().__init__()
        self.sliding_window_size = sliding_window_size
        self.confidence_level = confidence_level
        self.short_window_size = short_window_size
        self.n_one = 0
        self._reset()

    def _reset(self):
        self._sliding_window = collections.deque(maxlen=self.sliding_window_size)
        self._epsilon = math.sqrt(
            (math.log(1 / self.confidence_level)) / (2 * self.sliding_window_size)
        )
        self._u_max = 0
        self.n_one = 0
        if self.short_window_size is not None:
            self._short_window = collections.deque(maxlen=self.short_window_size)
            self._u_short_max = 0
            self._epsilon_s = math.sqrt(
                (math.log(1 / self.confidence_level)) / (2 * self.short_window_size)
            )

    def update(self, x):
        if self.drift_detected:
            self._drift_detected = False
            self._reset()

        self._sliding_window.append(x)
        self.n_one += x

        if len(self._sliding_window) == self.sliding_window_size:
            u = self.n_one / self.sliding_window_size
            self._u_max = u if (self._u_max < u) else self._u_max

            short_win_drift_status = False
            long_win_drift_status = False

            if self.short_window_size is not None:
                u_s = (
                    sum(
                        itertools.islice(
                            self._sliding_window,
                            self.sliding_window_size - self.short_window_size,
                            self.sliding_window_size,
                        )
                    )
                    / self.short_window_size
                )
                self._u_short_max = u_s if self._u_short_max < u_s else self._u_short_max
                short_win_drift_status = (
                    True if self._u_short_max - u_s > self._epsilon_s else False
                )

            long_win_drift_status = True if (self._u_max - u > self._epsilon) else False

            self._drift_detected = long_win_drift_status or short_win_drift_status
            self.n_one -= self._sliding_window.popleft()
