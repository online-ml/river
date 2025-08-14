from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import deque

from river import base


class _MDDMBase(base.BinaryDriftAndWarningDetector, ABC):
    """
    McDiarmid inequality detects a drift if:

        Δμ_w ≥ ε_w

    where:

        ε_w = sqrt( (1/2) * sum_{i=1}^{n} v_i^2 * ln(1/δ_w) )

    and:

        v_i = w_i / sum_{i=1}^{n} w_i

    Note:
        For a given weighting scheme, the denominator in the definition of `v_i`
        is computed once using the method `_calculate_sum_of_weights`,and remains constant.
        The value is cached as it is reused during the streaming process.

        Also, the sum of squared normalized weights (i.e., ∑ v_i²) is computed by the
        method `_cal_sum_of_sq_of_norm_weights`.
    """

    def __init__(
        self,
        sliding_window_size: int = 100,
        drift_confidence: float = 0.000001,
        warning_confidence: float = 0.000005,
    ) -> None:
        self.sliding_window_size = sliding_window_size
        self.drift_confidence = drift_confidence
        self.warning_confidence = warning_confidence
        self._reset()

    def _reset(self) -> None:
        """
        Reset the learning state of the model.
        """
        super()._reset()
        self._sliding_window: deque[int] = deque(maxlen=self.sliding_window_size)
        self._SUM_OF_WEIGHTS: float = self._calculate_sum_of_weights()
        sum_of_sq_of_norm_weights = self._cal_sum_of_sq_of_norm_weights()
        self._drift_epsilon: float = self._mcdiarmid_inequality_bound(
            self.drift_confidence, sum_of_sq_of_norm_weights
        )
        self._warning_epsilon: float = self._mcdiarmid_inequality_bound(
            self.warning_confidence, sum_of_sq_of_norm_weights
        )
        self._max_weighted_mean: float = 0.0

    def _mcdiarmid_inequality_bound(self, confidence: float, sum_of_sq_norm_weights) -> float:
        return math.sqrt(0.5 * sum_of_sq_norm_weights * math.log(1 / confidence))

    @abstractmethod
    def _calculate_sum_of_weights(self) -> float: ...

    @abstractmethod
    def _cal_sum_of_sq_of_norm_weights(self) -> float: ...

    def update(self, x: int) -> None:
        """Update the change detector with a single data point.

        Parameters
        ----------
        x
            This parameter indicates whether the last sample analyzed was
            correctly classified or not. 1 indicates correct prediction and
            0 indicates an error (miss-classification).

        Returns
        -------
        self

        """
        assert x in (0, 1), "Input must be binary (0 or 1)."
        if self.drift_detected:
            self._reset()

        self._sliding_window.append(0 if x == 1 else 1)

        if len(self._sliding_window) == self.sliding_window_size:
            current_weighted_mean: float = self._calculate_current_weighted_mean()
            self._max_weighted_mean = max(self._max_weighted_mean, current_weighted_mean)

            if self._max_weighted_mean - current_weighted_mean > self._drift_epsilon:
                self._warning_detected = False
                self._drift_detected = True
            elif self._max_weighted_mean - current_weighted_mean > self._warning_epsilon:
                self._warning_detected = True
                self._drift_detected = False

    @abstractmethod
    def _calculate_current_weighted_mean(self) -> float: ...


class MDDM_A(_MDDMBase):
    """
    McDiarmid Drift Detection Method using Arithmetic weighting (MDDM-A).

    MDDM-A is a drift detection method based on McDiarmid's inequality, which
    provides bounds on the deviation of a weighted mean in a sliding window
    of binary classification errors (0 for correct/normal values, 1 for error/failure).

    In MDDM-A, weights increase linearly from the oldest to the newest element
    in the window, controlled by a user-defined `difference` parameter. This
    places greater emphasis on more recent samples when detecting drift.

    **Input:** `x` is an entry in a stream of bits, where 1 indicates error/failure and 0
    represents correct/normal values.

    For example, if a classifier's prediction $y'$ is right or wrong w.r.t. the
    true target label $y$:

    - 0: Correct, $y=y'$

    - 1: Error, $y \\neq y'$

    A drift is detected when the current weighted mean significantly deviates
    (according to McDiarmid's inequality) from the maximum weighted mean observed so far.

    Parameters
    ----------
    sliding_window_size
        Number of recent binary values to store for computing the weighted mean.
    difference
        Weight increment per step from oldest to newest value. Higher values give more
        weight to recent samples.
    confidence
        Confidence level for drift detection. Smaller values make the detector more sensitive.
    warning_confidence
        Confidence level for triggering a warning zone before drift is detected.

    Examples
    --------
    >>> import random
    >>> from river import drift
    >>>
    >>> rng = random.Random(42)
    >>> mddm_a = drift.binary.MDDM_A()
    >>>
    >>> # Simulate a data stream where the first 1000 instances have balanced distribution
    >>> data_stream = rng.choices([0, 1], k=1000)
    >>> # Increase the probability of errors in the next 1000 instances
    >>> data_stream += rng.choices([0, 1], k=1000, weights=[0.3, 0.7])
    >>>
    >>> print_warning = True
    >>> for i, x in enumerate(data_stream):
    ...     mddm_a.update(x)
    ...     if mddm_a.warning_detected and print_warning:
    ...         print(f"Warning detected at index {i}")
    ...         print_warning = False
    ...     if mddm_a.drift_detected:
    ...         print(f"Drift detected at index {i}")
    ...         print_warning = True
    Warning detected at index 1107
    Drift detected at index 1120

    References
    ----------
    [^1]: Pesaranghader, A., Viktor, H.L. and Paquet, E., 2018.
    McDiarmid drift detection methods for evolving data streams.
    In 2018 International Joint Conference on Neural Networks (IJCNN) (pp. 1-9). IEEE.
    """

    def __init__(
        self,
        sliding_window_size: int = 100,
        difference: float = 0.01,
        confidence: float = 0.000001,
        warning_confidence: float = 0.000005,
    ) -> None:
        self.difference: float = difference
        super().__init__(sliding_window_size, confidence, warning_confidence)

    def _calculate_sum_of_weights(self) -> float:
        return sum(1 + i * self.difference for i in range(self.sliding_window_size))

    def _cal_sum_of_sq_of_norm_weights(self) -> float:
        return sum(
            ((1 + i * self.difference) / self._SUM_OF_WEIGHTS) ** 2
            for i in range(self.sliding_window_size)
        )

    def _calculate_current_weighted_mean(self) -> float:
        win_sum: float = sum(
            self._sliding_window[i] * (1 + i * self.difference)
            for i in range(len(self._sliding_window))
        )
        return win_sum / self._SUM_OF_WEIGHTS


class MDDM_E(_MDDMBase):
    """
    McDiarmid Drift Detection Method using Exponential (Euler) weighting (MDDM-E).

    MDDM-E is a drift detection method based on McDiarmid's inequality, which
    provides bounds on the deviation of a weighted mean in a sliding window
    of binary classification errors (0 for correct/normal values, 1 for error/failure).

    In MDDM-E, weights increase exponentially from the oldest to the newest element
    in the window, controlled by a user-defined `lambda_val` parameter. This
    places greater emphasis on more recent samples when detecting drift.

    **Input:** `x` is an entry in a stream of bits, where 1 indicates error/failure and 0
    represents correct/normal values.

    For example, if a classifier's prediction $y'$ is right or wrong w.r.t. the
    true target label $y$:

    - 0: Correct, $y=y'$

    - 1: Error, $y \\neq y'$

    A drift is detected when the current weighted mean significantly deviates
    (according to McDiarmid's inequality) from the maximum weighted mean observed so far.

    Parameters
    ----------
    sliding_window_size
        Number of recent binary values to store for computing the weighted mean.
    lambda_val
        Exponential growth rate for weights applied from oldest to newest value.
    confidence
        Confidence level for drift detection. Smaller values make the detector more sensitive.
    warning_confidence
        Confidence level for triggering a warning zone before drift is detected.

    Examples
    --------
    >>> import random
    >>> from river import drift
    >>>
    >>> rng = random.Random(42)
    >>> mddm_e = drift.binary.MDDM_E()
    >>>
    >>> # Simulate a data stream where the first 1000 instances have balanced distribution
    >>> data_stream = rng.choices([0, 1], k=1000)
    >>> # Increase the probability of errors in the next 1000 instances
    >>> data_stream += rng.choices([0, 1], k=1000, weights=[0.3, 0.7])
    >>>
    >>> print_warning = True
    >>> for i, x in enumerate(data_stream):
    ...     mddm_e.update(x)
    ...     if mddm_e.warning_detected and print_warning:
    ...         print(f"Warning detected at index {i}")
    ...         print_warning = False
    ...     if mddm_e.drift_detected:
    ...         print(f"Drift detected at index {i}")
    ...         print_warning = True
    Warning detected at index 1106
    Drift detected at index 1110

    References
    ----------
    [^1]: Pesaranghader, A., Viktor, H.L. and Paquet, E., 2018.
    McDiarmid drift detection methods for evolving data streams.
    In 2018 International Joint Conference on Neural Networks (IJCNN) (pp. 1-9). IEEE.
    """

    def __init__(
        self,
        sliding_window_size: int = 100,
        lambda_val: float = 0.01,
        confidence: float = 0.000001,
        warning_confidence: float = 0.000005,
    ) -> None:
        self.lambda_val: float = lambda_val
        super().__init__(sliding_window_size, confidence, warning_confidence)

    def _calculate_sum_of_weights(self) -> float:
        sum_exp: float = 0.0
        r: float = 1.0
        ratio: float = math.exp(self.lambda_val)
        for _ in range(self.sliding_window_size):
            sum_exp += r
            r *= ratio
        return sum_exp

    def _cal_sum_of_sq_of_norm_weights(self) -> float:
        bound_sum: float = 0.0
        r: float = 1.0
        ratio: float = math.exp(self.lambda_val)
        for _ in range(self.sliding_window_size):
            bound_sum += (r / self._SUM_OF_WEIGHTS) ** 2
            r *= ratio
        return bound_sum

    def _calculate_current_weighted_mean(self) -> float:
        win_sum: float = 0.0
        r: float = 1.0
        ratio: float = math.exp(self.lambda_val)
        for i in range(len(self._sliding_window)):
            win_sum += self._sliding_window[i] * r
            r *= ratio
        return win_sum / self._SUM_OF_WEIGHTS


class MDDM_G(_MDDMBase):
    """
    McDiarmid Drift Detection Method using Geometric weighting (MDDM-G).

    MDDM-G is a drift detection method based on McDiarmid's inequality, which
    provides bounds on the deviation of a weighted mean in a sliding window
    of binary classification errors (0 for correct/normal values, 1 for error/failure).

    In MDDM-G, weights increase geometrically from the oldest to the newest element
    in the window, controlled by a user-defined `ratio` parameter. This
    places greater emphasis on more recent samples when detecting drift.

    **Input:** `x` is an entry in a stream of bits, where 1 indicates error/failure and 0
    represents correct/normal values.

    For example, if a classifier's prediction $y'$ is right or wrong w.r.t. the
    true target label $y$:

    - 0: Correct, $y=y'$

    - 1: Error, $y \\neq y'$

    A drift is detected when the current weighted mean significantly deviates
    (according to McDiarmid's inequality) from the maximum weighted mean observed so far.

    Parameters
    ----------
    sliding_window_size
        Number of recent binary values to store for computing the weighted mean.
    ratio
        Multiplicative factor for weights applied from oldest to newest value.
    confidence
        Confidence level for drift detection. Smaller values make the detector more sensitive.
    warning_confidence
        Confidence level for triggering a warning zone before drift is detected.

    Examples
    --------
    >>> import random
    >>> from river import drift
    >>>
    >>> rng = random.Random(42)
    >>> mddm_g = drift.binary.MDDM_G()
    >>>
    >>> # Simulate a data stream where the first 1000 instances have balanced distribution
    >>> data_stream = rng.choices([0, 1], k=1000)
    >>> # Increase the probability of errors in the next 1000 instances
    >>> data_stream += rng.choices([0, 1], k=1000, weights=[0.3, 0.7])
    >>>
    >>> print_warning = True
    >>> for i, x in enumerate(data_stream):
    ...     mddm_g.update(x)
    ...     if mddm_g.warning_detected and print_warning:
    ...         print(f"Warning detected at index {i}")
    ...         print_warning = False
    ...     if mddm_g.drift_detected:
    ...         print(f"Drift detected at index {i}")
    ...         print_warning = True
    Warning detected at index 1106
    Drift detected at index 1110

    References
    ----------
    [^1]: Pesaranghader, A., Viktor, H.L. and Paquet, E., 2018.
    McDiarmid drift detection methods for evolving data streams.
    In 2018 International Joint Conference on Neural Networks (IJCNN) (pp. 1-9). IEEE.
    """

    def __init__(
        self,
        sliding_window_size: int = 100,
        ratio: float = 1.01,
        confidence: float = 0.000001,
        warning_confidence: float = 0.000005,
    ) -> None:
        self.ratio: float = ratio
        super().__init__(sliding_window_size, confidence, warning_confidence)

    def _calculate_sum_of_weights(self) -> float:
        sum_exp: float = 0.0
        r: float = self.ratio
        for _ in range(self.sliding_window_size):
            sum_exp += r
            r *= self.ratio
        return sum_exp

    def _cal_sum_of_sq_of_norm_weights(self) -> float:
        bound_sum: float = 0.0
        r: float = self.ratio
        for _ in range(self.sliding_window_size):
            bound_sum += (r / self._SUM_OF_WEIGHTS) ** 2
            r *= self.ratio
        return bound_sum

    def _calculate_current_weighted_mean(self) -> float:
        win_sum: float = 0.0
        r: float = self.ratio
        for i in range(len(self._sliding_window)):
            win_sum += self._sliding_window[i] * r
            r *= self.ratio
        return win_sum / self._SUM_OF_WEIGHTS
