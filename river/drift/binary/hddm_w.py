from __future__ import annotations

import copy
import math

from river import base, stats


class HDDM_W(base.BinaryDriftAndWarningDetector):
    """Drift Detection Method based on Hoeffding's bounds with moving weighted average-test.

    HDDM_W is an online drift detection method based on McDiarmid's bounds. HDDM_W uses the
    Exponentially Weighted Moving Average (EWMA) statistic as estimator.

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
    lambda_val
        The weight given to recent data. Smaller values mean less weight given to recent data.
    two_sided_test
        If True, will monitor error increments and decrements (two-sided). By default will only
        monitor increments (one-sided).

    Examples
    --------
    >>> import random
    >>> from river import drift

    >>> rng = random.Random(42)
    >>> hddm_w = drift.binary.HDDM_W()

    >>> # Simulate a data stream where the first 1000 instances come from a uniform distribution
    >>> # of 1's and 0's
    >>> data_stream = rng.choices([0, 1], k=1000)
    >>> # Increase the probability of 1's appearing in the next 1000 instances
    >>> data_stream = data_stream + rng.choices([0, 1], k=1000, weights=[0.3, 0.7])

    >>> print_warning = True
    >>> # Update drift detector and verify if change is detected
    >>> for i, x in enumerate(data_stream):
    ...     _ = hddm_w.update(x)
    ...     if hddm_w.warning_detected and print_warning:
    ...         print(f"Warning detected at index {i}")
    ...         print_warning = False
    ...     if hddm_w.drift_detected:
    ...         print(f"Change detected at index {i}")
    ...         print_warning = True
    Warning detected at index 451
    Change detected at index 1077

     References
     ----------
     [^1]: Frías-Blanco I, del Campo-Ávila J, Ramos-Jimenez G, et al. Online and non-parametric drift
     detection methods based on Hoeffding’s bounds. IEEE Transactions on Knowledge and Data
     Engineering, 2014, 27(3): 810-823.
     [^2]: Albert Bifet, Geoff Holmes, Richard Kirkby, Bernhard Pfahringer. MOA: Massive Online Analysis;
     Journal of Machine Learning Research 11: 1601-1604, 2010.

    """

    def __init__(
        self,
        drift_confidence=0.001,
        warning_confidence=0.005,
        lambda_val=0.05,
        two_sided_test=False,
    ):
        super().__init__()
        self.drift_confidence = drift_confidence
        self.warning_confidence = warning_confidence
        self.lambda_val = lambda_val
        self.two_sided_test = two_sided_test

        self._reset()

    def _reset(self):
        super()._reset()
        self._total = SampleInfo(self.lambda_val)
        self._s1_decr = SampleInfo(self.lambda_val)
        self._s1_incr = SampleInfo(self.lambda_val)
        self._s2_decr = SampleInfo(self.lambda_val)
        self._s2_incr = SampleInfo(self.lambda_val)
        self._incr_cutpoint = float("inf")
        self._decr_cutpoint = -float("inf")

    def _mcdiarmid_bound(self, ibc: float, confidence: float):
        return math.sqrt(ibc * math.log(1 / confidence) / 2)

    def update(self, x):
        """Update the change detector with a single data point.

        Parameters
        ----------
        x
            This parameter indicates whether the last sample analyzed was
            correctly classified or not. 1 indicates an error (miss-classification).

        Returns
        -------
        self

        """

        if self.drift_detected:
            self._reset()

        self._total.update(x)

        self._update_incr_stats(x, self.drift_confidence)
        if self._detect_mean_incr(self.drift_confidence):
            self._warning_detected = False
            self._drift_detected = True
        elif self._detect_mean_incr(self.warning_confidence):
            self._warning_detected = True
            self._drift_detected = False
        else:
            self._warning_detected = False
            self._drift_detected = False

        self._update_decr_stats(x, self.drift_confidence)
        if self.two_sided_test:
            if self._detect_mean_decr(self.drift_confidence):
                self._drift_detected = True
            elif self._detect_mean_decr(self.warning_confidence):
                self._warning_detected = True

        return self

    def _has_mean_changed(
        self, sample1: SampleInfo, sample2: SampleInfo, confidence: float
    ) -> bool:
        if not (sample1._is_init and sample2._is_init):
            return False
        ibc_sum = sample1.ibc + sample2.ibc
        bound = self._mcdiarmid_bound(ibc_sum, confidence)

        return sample2.ewma - sample1.ewma > bound

    def _detect_mean_incr(self, confidence: float) -> bool:
        return self._has_mean_changed(self._s1_incr, self._s2_incr, confidence)

    def _detect_mean_decr(self, confidence: float) -> bool:
        return self._has_mean_changed(self._s2_decr, self._s1_decr, confidence)

    def _update_incr_stats(self, x, confidence):
        eps = self._mcdiarmid_bound(self._total.ibc, confidence)

        if self._total.ewma + eps < self._incr_cutpoint:
            self._incr_cutpoint = self._total.ewma + eps
            self._s1_incr = copy.deepcopy(self._total)
            self._s2_incr = SampleInfo(self.lambda_val)
        else:
            self._s2_incr.update(x)

    def _update_decr_stats(self, x, confidence):
        eps = self._mcdiarmid_bound(self._total.ibc, confidence)

        if self._total.ewma - eps > self._decr_cutpoint:
            self._decr_cutpoint = self._total.ewma - eps
            self._s1_decr = copy.deepcopy(self._total)
            self._s2_decr = SampleInfo(self.lambda_val)
        else:
            self._s2_decr.update(x)


class SampleInfo:
    def __init__(self, lambd: float):
        self._ewma = stats.EWMean(lambd)
        self._is_init = False
        # Independent bound condition
        self._ibc = 1.0
        self._lambd_sq = lambd * lambd
        self._c_lambd_sq = (1 - lambd) ** 2

    def update(self, x):
        self._ewma.update(x)
        self._is_init = True
        self._ibc = self._lambd_sq + self._c_lambd_sq * self._ibc

    @property
    def ewma(self):
        return self._ewma.get()

    @property
    def ibc(self):
        return self._ibc

    @property
    def is_init(self):
        return self._is_init
