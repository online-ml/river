from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import deque

from river import base


class _MDDMBase(base.BinaryDriftAndWarningDetector, ABC):
    def __init__(
        self,
        sliding_window_size: int = 100,
        drift_confidence: float = 0.000001,
        warning_confidence: float = 0.000005,
    ) -> None:
        """
        Initialize the MDDM base class with specified parameters.

        Args:
            sliding_window_size (int, optional): Size of the sliding window (default: 100).
            drift_confidence (float, optional): Confidence level for drift detection (default: 0.000001).
            warning_confidence (float, optional): Confidence level for warning zone detection (default: 0.000005).
        """
        self.sliding_window_size = sliding_window_size
        self.drift_confidence = drift_confidence
        self.warning_confidence = warning_confidence
        self._reset()

    def _reset(self) -> None:
        """
        Reset the learning state of the model.
        """
        super()._reset()
        self._sliding_window: deque = deque(maxlen=self.sliding_window_size)
        self._SUM_OF_WEIGHTS = self._calculate_sum_of_weights()
        self._drift_epsilon = self._mcdiarmid_inequality_bound(self.drift_confidence)
        self._warning_epsilon = self._mcdiarmid_inequality_bound(self.warning_confidence)
        self._max_weighted_mean = 0.0

    def _mcdiarmid_inequality_bound(self, confidence: float) -> float:
        return math.sqrt(
            0.5 * self._calculate_sum_of_normalized_weights() * math.log(1 / confidence)
        )

    @abstractmethod
    def _calculate_sum_of_weights(self) -> float: ...

    @abstractmethod
    def _calculate_sum_of_normalized_weights(self) -> float: ...

    def update(self, x: int) -> None:
        if self.drift_detected:
            self._reset()

        self._sliding_window.append(1 if x == 0 else 0)

        if len(self._sliding_window) == self.sliding_window_size:
            current_weighted_mean = self._calculate_current_weighted_mean()
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
    def __init__(
        self,
        sliding_win_size: int = 100,
        difference: float = 0.01,
        confidence: float = 0.000001,
        warning_confidence: float = 0.000005,
    ) -> None:
        super().__init__(sliding_win_size, confidence, warning_confidence)
        self.difference = difference

    def _calculate_sum_of_weights(self) -> float:
        return sum(1 + i * self.difference for i in range(self.sliding_window_size))

    def _calculate_sum_of_normalized_weights(self):
        return sum(
            ((1 + i * self.difference) / self._SUM_OF_WEIGHTS) ** 2
            for i in range(self.sliding_window_size)
        )

    def _calculate_current_weighted_mean(self) -> float:
        """
        Calculate the weighted mean for Arithmetic weighting method.

        Returns:
            float: Calculated weighted mean.
        """
        win_sum = sum(
            self._sliding_window[i] * (1 + i * self.difference)
            for i in range(len(self._sliding_window))
        )
        return win_sum / self._SUM_OF_WEIGHTS


class MDDM_E(_MDDMBase):
    def __init__(
        self,
        sliding_win_size: int = 100,
        lambda_val: float = 0.01,
        confidence: float = 0.000001,
        warning_confidence: float = 0.000005,
    ) -> None:
        super().__init__(sliding_win_size, confidence, warning_confidence)
        self.lambda_val = lambda_val

    def _calculate_sum_of_weights(self) -> float:
        sum_exp = 0.0
        r = 1.0
        ratio = math.exp(self.lambda_val)
        for _ in range(self.sliding_window_size):
            sum_exp += r
            r *= ratio
        return sum_exp

    def _calculate_sum_of_normalized_weights(self) -> float:
        bound_sum = 0.0
        r = 1.0
        ratio = math.exp(self.lambda_val)
        for _ in range(self.sliding_window_size):
            bound_sum += (r / self._SUM_OF_WEIGHTS) ** 2
            r *= ratio
        return bound_sum

    def _calculate_current_weighted_mean(self) -> float:
        """
        Calculate the weighted mean for Euler Weighting method.

        Returns:
            float: Calculated weighted mean.
        """
        win_sum = 0.0
        r = 1.0
        ratio = math.exp(self.lambda_val)
        for i in range(len(self._sliding_window)):
            win_sum += self._sliding_window[i] * r
            r *= ratio
        return win_sum / self._SUM_OF_WEIGHTS


class MDDM_G(_MDDMBase):
    def __init__(
        self,
        sliding_win_size: int = 100,
        ratio: float = 1.01,
        confidence: float = 0.000001,
        warning_confidence: float = 0.000005,
    ) -> None:
        super().__init__(sliding_win_size, confidence, warning_confidence)
        self.ratio = ratio

    def _calculate_sum_of_weights(self):
        sum_exp = 0.0
        r = self.ratio
        for _ in range(self.sliding_window_size):
            sum_exp += r
            r *= self.ratio
        return sum_exp

    def _calculate_sum_of_normalized_weights(self) -> float:
        bound_sum = 0.0
        r = self.ratio
        for _ in range(self.sliding_window_size):
            bound_sum += (r / self._SUM_OF_WEIGHTS) ** 2
            r *= self.ratio
        return bound_sum

    def _calculate_current_weighted_mean(self) -> float:
        """
        Calculate the weighted mean for Geometric Weighting method.

        Returns:
            float: Calculated weighted mean.
        """
        win_sum = 0.0
        r = self.ratio
        for i in range(len(self._sliding_window)):
            win_sum += self._sliding_window[i] * r
            r *= self.ratio
        return win_sum / self._SUM_OF_WEIGHTS
