from math import log, sqrt

from river.base import DriftDetector


class HDDM_A(DriftDetector):
    r"""Drift Detection Method based on Hoeffding’s bounds with moving average-test.

    HDDM_A is a drift detection method based on the Hoeffding’s inequality.
    HDDM_A uses the average as estimator. It receives as input a stream of real
    values and returns the estimated status of the stream: STABLE, WARNING or
    DRIFT.

    **Input:** `value` must be a binary signal, where 0 indicates error.
    For example, if a classifier's prediction $y'$ is right or wrong w.r.t the
    true target label $y$:

    - 0: Correct, $y=y'$

    - 1: Error, $y \neq y'$

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
    >>> import numpy as np
    >>> from river.drift import HDDM_A
    >>> np.random.seed(12345)

    >>> hddm_a = HDDM_A()

    >>> # Simulate a data stream as a normal distribution of 1's and 0's
    >>> data_stream = np.random.randint(2, size=2000)
    >>> # Change the data distribution from index 999 to 1500, simulating an
    >>> # increase in error rate (1 indicates error)
    >>> data_stream[999:1500] = 1

    >>> # Update drift detector and verify if change is detected
    >>> for i, val in enumerate(data_stream):
    ...     in_drift, in_warning = hddm_a.update(val)
    ...     if in_drift:
    ...         print(f"Change detected at index {i}, input value: {val}")
    Change detected at index 1013, input value: 1

    References
    ----------
    [^1]: Frías-Blanco I, del Campo-Ávila J, Ramos-Jimenez G, et al. Online and non-parametric drift detection methods based on Hoeffding’s bounds. IEEE Transactions on Knowledge and Data Engineering, 2014, 27(3): 810-823.
    [^2]: Albert Bifet, Geoff Holmes, Richard Kirkby, Bernhard Pfahringer. MOA: Massive Online Analysis; Journal of Machine Learning Research 11: 1601-1604, 2010.

    """

    def __init__(self, drift_confidence=0.001, warning_confidence=0.005, two_sided_test=False):
        super().__init__()
        super().reset()
        self.n_min = 0
        self.c_min = 0
        self.total_n = 0
        self.total_c = 0
        self.n_max = 0
        self.c_max = 0
        self.n_estimation = 0
        self.c_estimation = 0
        self.estimation = None
        self.delay = None

        self.drift_confidence = drift_confidence
        self.warning_confidence = warning_confidence
        self.two_sided_test = two_sided_test

    def update(self, value) -> tuple:
        """Update the change detector with a single data point.

        Parameters
        ----------
        value
            This parameter indicates whether the last sample analyzed was correctly classified or
            not. 1 indicates an error (miss-classification).

        Returns
        -------
        A tuple (drift, warning) where its elements indicate if a drift or a warning is detected.

        """
        self.total_n += 1
        self.total_c += value
        if self.n_min == 0:
            self.n_min = self.total_n
            self.c_min = self.total_c
        if self.n_max == 0:
            self.n_max = self.total_n
            self.c_max = self.total_c

        cota = sqrt(1.0 / (2 * self.n_min) * log(1.0 / self.drift_confidence))
        cota1 = sqrt(1.0 / (2 * self.total_n) * log(1.0 / self.drift_confidence))

        if self.c_min / self.n_min + cota >= self.total_c / self.total_n + cota1:
            self.c_min = self.total_c
            self.n_min = self.total_n

        cota = sqrt(1.0 / (2 * self.n_max) * log(1.0 / self.drift_confidence))
        if self.c_max / self.n_max - cota <= self.total_c / self.total_n - cota1:
            self.c_max = self.total_c
            self.n_max = self.total_n

        if self._mean_incr(
            self.c_min, self.n_min, self.total_c, self.total_n, self.drift_confidence
        ):
            self.n_estimation = self.total_n - self.n_min
            self.c_estimation = self.total_c - self.c_min
            self.n_min = self.n_max = self.total_n = 0
            self.c_min = self.c_max = self.total_c = 0
            self._in_concept_change = True
            self._in_warning_zone = False
        elif self._mean_incr(
            self.c_min, self.n_min, self.total_c, self.total_n, self.warning_confidence
        ):
            self._in_concept_change = False
            self._in_warning_zone = True
        else:
            self._in_concept_change = False
            self._in_warning_zone = False

        if self.two_sided_test and self._mean_decr(
            self.c_max, self.n_max, self.total_c, self.total_n
        ):
            self.n_estimation = self.total_n - self.n_max
            self.c_estimation = self.total_c - self.c_max
            self.n_min = self.n_max = self.total_n = 0
            self.c_min = self.c_max = self.total_c = 0

        self._update_estimations()

        return self._in_concept_change, self._in_warning_zone

    @staticmethod
    def _mean_incr(c_min, n_min, total_c, total_n, confidence):
        if n_min == total_n:
            return False

        m = (total_n - n_min) / n_min * (1.0 / total_n)
        cota = sqrt(m / 2 * log(2.0 / confidence))
        return total_c / total_n - c_min / n_min >= cota

    def _mean_decr(self, c_max, n_max, total_c, total_n):
        if n_max == total_n:
            return False

        m = (total_n - n_max) / n_max * (1.0 / total_n)
        cota = sqrt(m / 2 * log(2.0 / self.drift_confidence))
        return c_max / n_max - total_c / total_n >= cota

    def reset(self):
        """Reset the change detector."""
        super().reset()
        self.n_min = 0
        self.c_min = 0
        self.total_n = 0
        self.total_c = 0
        self.n_max = 0
        self.c_max = 0
        self.c_estimation = 0
        self.n_estimation = 0

    def _update_estimations(self):
        if self.total_n >= self.n_estimation:
            self.c_estimation = self.n_estimation = 0
            self.estimation = self.total_c / self.total_n
            self.delay = self.total_n
        else:
            self.estimation = self.c_estimation / self.n_estimation
            self.delay = self.n_estimation
