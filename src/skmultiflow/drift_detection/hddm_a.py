from math import log, sqrt

from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector


class HDDM_A(BaseDriftDetector):
    """
    Drift Detection Method based on Hoeffding’s bounds with moving average-test.

    Parameters
    ----------
    drift_confidence : float (default=0.001)
        Confidence to the drift

    warning_confidence : float (default=0.005)
        Confidence to the warning

    two_side_option : bool (default=True)
        Option to monitor error increments and decrements (two-sided) or only increments
        (one-sided)

    Notes
    -----
    HDDM_A [1]_ is a drift detection method based on the Hoeffding’s inequality. HDDM_A uses
    the average as estimator. It receives as input a stream of real values and
    returns the estimated status of the stream: STABLE, WARNING or DRIFT.

    Implementation based on MOA [2]_.

    References
    ----------
    .. [1] Frías-Blanco I, del Campo-Ávila J, Ramos-Jimenez G, et al.
       Online and non-parametric drift detection methods based on Hoeffding’s bounds.
       IEEE Transactions on Knowledge and Data Engineering, 2014, 27(3): 810-823.

    .. [2] Albert Bifet, Geoff Holmes, Richard Kirkby, Bernhard Pfahringer.
       MOA: Massive Online Analysis; Journal of Machine Learning Research 11: 1601-1604, 2010.

    Examples
    --------
    >>> # Imports
    >>> import numpy as np
    >>> from skmultiflow.drift_detection.hddm_a import HDDM_A
    >>> hddm_a = HDDM_A()
    >>> # Simulating a data stream as a normal distribution of 1's and 0's
    >>> data_stream = np.random.randint(2, size=2000)
    >>> # Changing the data concept from index 999 to 1500, simulating an
    >>> # increase in error rate
    >>> for i in range(999, 1500):
    ...     data_stream[i] = 0
    >>> # Adding stream elements to HDDM_A and verifying if drift occurred
    >>> for i in range(2000):
    ...     hddm_a.add_element(data_stream[i])
    ...     if hddm_a.detected_warning_zone():
    ...         print("Warning zone has been detected in data: {} - of index: {}"
    ...                .format(data_stream[i], i))
    ...     if hddm_a.detected_change():
    ...         print("Change has been detected in data: {} - of index: {}"
    ...                .format(data_stream[i], i))

    """

    def __init__(self, drift_confidence=0.001, warning_confidence=0.005, two_side_option=True):
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

        self.drift_confidence = drift_confidence
        self.warning_confidence = warning_confidence
        self.two_side_option = two_side_option

    def add_element(self, prediction):
        """ Add a new element to the statistics

        Parameters
        ----------
        prediction: int (either 0 or 1)
            This parameter indicates whether the last sample analyzed was
            correctly classified or not. 1 indicates an error (miss-classification).

        Notes
        -----
        After calling this method, to verify if change was detected or if
        the learner is in the warning zone, one should call the super method
        detected_change, which returns True if concept drift was detected and
        False otherwise.

        """
        self.total_n += 1
        self.total_c += prediction
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
                self.c_min,
                self.n_min,
                self.total_c,
                self.total_n,
                self.drift_confidence):
            self.n_estimation = self.total_n - self.n_min
            self.c_estimation = self.total_c - self.c_min
            self.n_min = self.n_max = self.total_n = 0
            self.c_min = self.c_max = self.total_c = 0
            self.in_concept_change = True
            self.in_warning_zone = False
        elif self._mean_incr(self.c_min, self.n_min, self.total_c, self.total_n,
                             self.warning_confidence):
            self.in_concept_change = False
            self.in_warning_zone = True
        else:
            self.in_concept_change = False
            self.in_warning_zone = False

        if self.two_side_option and self._mean_decr(
                self.c_max, self.n_max, self.total_c, self.total_n):
            self.n_estimation = self.total_n - self.n_max
            self.c_estimation = self.total_c - self.c_max
            self.n_min = self.n_max = self.total_n = 0
            self.c_min = self.c_max = self.total_c = 0

        self._update_estimations()

    def _mean_incr(self, c_min, n_min, total_c, total_n, confidence):
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
        """ reset

        Resets the change detector parameters.

        """
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
        """ update_estimations

        Update the length estimation and delay.

        """
        if self.total_n >= self.n_estimation:
            self.c_estimation = self.n_estimation = 0
            self.estimation = self.total_c / self.total_n
            self.delay = self.total_n
        else:
            self.estimation = self.c_estimation / self.n_estimation
            self.delay = self.n_estimation
