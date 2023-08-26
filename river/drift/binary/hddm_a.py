from __future__ import annotations

import math

from river import base, stats


class HDDM_A(base.BinaryDriftAndWarningDetector):
    """Drift Detection Method based on Hoeffding's bounds with moving average-test.

    HDDM_A is a drift detection method based on the Hoeffding's inequality which uses
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
    drift_confidence
        Confidence to the drift
    warning_confidence
        Confidence to the warning
    two_sided_test
        If `True`, will monitor error increments and decrements (two-sided). By default will only
        monitor increments (one-sided).

    Examples
    --------
    >>> import random
    >>> from river import drift

    >>> rng = random.Random(42)
    >>> hddm_a = drift.binary.HDDM_A()

    >>> # Simulate a data stream where the first 1000 instances come from a uniform distribution
    >>> # of 1's and 0's
    >>> data_stream = rng.choices([0, 1], k=1000)
    >>> # Increase the probability of 1's appearing in the next 1000 instances
    >>> data_stream = data_stream + rng.choices([0, 1], k=1000, weights=[0.3, 0.7])

    >>> print_warning = True
    >>> # Update drift detector and verify if change is detected
    >>> for i, x in enumerate(data_stream):
    ...     _ = hddm_a.update(x)
    ...     if hddm_a.warning_detected and print_warning:
    ...         print(f"Warning detected at index {i}")
    ...         print_warning = False
    ...     if hddm_a.drift_detected:
    ...         print(f"Change detected at index {i}")
    ...         print_warning = True
    Warning detected at index 451
    Change detected at index 1206

    References
    ----------
    [^1]: Frías-Blanco I, del Campo-Ávila J, Ramos-Jimenez G, et al. Online and non-parametric drift detection
    methods based on Hoeffding's bounds. IEEE Transactions on Knowledge and Data Engineering, 2014, 27(3): 810-823.
    [^2]: Albert Bifet, Geoff Holmes, Richard Kirkby, Bernhard Pfahringer. MOA: Massive Online Analysis; Journal
    of Machine Learning Research 11: 1601-1604, 2010.

    """

    def __init__(self, drift_confidence=0.001, warning_confidence=0.005, two_sided_test=False):
        super().__init__()
        self.drift_confidence = drift_confidence
        self.warning_confidence = warning_confidence
        self.two_sided_test = two_sided_test
        self._reset()

    def _reset(self):
        super()._reset()

        # To check if the global mean increased
        self._x_min = stats.Mean()
        # To check if the global mean decreased
        self._x_max = stats.Mean()
        # Global mean
        self._z = stats.Mean()

    def _hoeffding_bound(self, n):
        return math.sqrt(1.0 / (2 * n) * math.log(1.0 / self.drift_confidence))

    def update(self, x):
        """Update the change detector with a single data point.

        Parameters
        ----------
        value
            This parameter indicates whether the last sample analyzed was correctly classified or
            not. 1 indicates an error (miss-classification).

        Returns
        -------
        self

        """

        if self.drift_detected:
            self._reset()

        self._z.update(x)
        if self._x_min.n == 0:
            self._x_min = self._z.clone(include_attributes=True)
        if self._x_max.n == 0:
            self._x_max = self._z.clone(include_attributes=True)

        # Bound the data
        eps_z = self._hoeffding_bound(self._z.n)
        eps_x = self._hoeffding_bound(self._x_min.n)
        # Update the cut point for tracking mean increase
        if self._x_min.get() + eps_x >= self._z.get() + eps_z:
            self._x_min = self._z.clone(include_attributes=True)

        eps_x = self._hoeffding_bound(self._x_max.n)
        # Update the cut point for tracking mean decrease
        if self._x_max.get() - eps_x <= self._z.get() - eps_z:
            self._x_max = self._z.clone(include_attributes=True)

        if self._mean_incr(self.drift_confidence):
            self._warning_detected = False
            self._drift_detected = True
        elif self._mean_incr(self.warning_confidence):
            self._warning_detected = True
            self._drift_detected = False
        else:
            self._warning_detected = False
            self._drift_detected = False

        if self.two_sided_test:
            if self._mean_decr(self.drift_confidence):
                self._drift_detected = True
            elif self._mean_decr(self.warning_confidence):
                self._warning_detected = True

        return self

    # Check if the global mean increased
    def _mean_incr(self, confidence: float):
        if self._x_min.n == self._z.n:
            return False

        m = (self._z.n - self._x_min.n) / self._x_min.n * (1.0 / self._z.n)
        eps = math.sqrt(m / 2 * math.log(2.0 / confidence))
        return self._z.get() - self._x_min.get() >= eps

    # Check if the global mean decreased
    def _mean_decr(self, confidence: float):
        if self._x_max.n == self._z.n:
            return False

        m = (self._z.n - self._x_max.n) / self._x_max.n * (1.0 / self._z.n)
        eps = math.sqrt(m / 2 * math.log(2.0 / confidence))
        return self._x_max.get() - self._z.get() >= eps
