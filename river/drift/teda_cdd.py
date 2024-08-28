from __future__ import annotations

import math

from river.base import DriftDetector


class TEDACDD(DriftDetector):
    """ """

    def __init__(self, m=3.0, alpha=0.95, jaccard_threshold=0.8):
        super().__init__()

        self.m = m
        self.alpha = alpha
        self.jaccard_threshold = jaccard_threshold

        self._reference_model_n = 0.0
        self._reference_model_mean = 0.0
        self._reference_model_var = 0.0

        self._evolving_model_n = 0.0
        self._evolving_model_mean = 0.0
        self._evolving_model_var = 0.0

        self.__min_samples = math.ceil(1.0 / (1.0 - self.alpha))

        if self.m <= 0.0:
            raise ValueError("Parameter `m` must be greater than 0")
        if self.alpha <= 0.0 or self.alpha >= 1.0:
            raise ValueError("Parameter `alpha` must be between 0 and 1 exclusively")

    @property
    def reference_model_n(self):
        """Reference model number of samples"""
        return self._reference_model_n

    @property
    def reference_model_mean(self):
        """Reference model mean of samples"""
        return self._reference_model_mean

    @property
    def reference_model_var(self):
        """Reference model variance of samples"""
        return self._reference_model_var

    @property
    def evolving_model_n(self):
        """Evolving model number of samples"""
        return self._evolving_model_n

    @property
    def evolving_model_mean(self):
        """Evolving model mean of samples"""
        return self._evolving_model_mean

    @property
    def evolving_model_var(self):
        """Evolving model variance of samples"""
        return self._evolving_model_var

    @property
    def jaccard_index(self):
        drift_distance = abs(self._reference_model_mean - self._evolving_model_mean)
        reference_model_radius = math.sqrt(self._reference_model_var) * self.m
        evolving_model_radius = math.sqrt(self._evolving_model_var) * self.m
        return self._jaccard_index(reference_model_radius, evolving_model_radius, drift_distance)

    @property
    def min_samples(self):
        return self.__min_samples

    def _reset(self):
        """Reset the detector's state replacing the reference model by the evoling model."""
        super()._reset()
        self._replace_reference_model()

    def update(self, x):
        _drift_detected = False

        reference_model_teda_score = self._reference_model_teda_score(x)
        reference_model_teda_threshold = self._reference_model_teda_threshold()
        if (
            reference_model_teda_score <= reference_model_teda_threshold
            or self.reference_model_n <= self.__min_samples
        ):
            self._update_reference_model(x)

        self._update_evolving_model(x)

        drift_distance = abs(self._reference_model_mean - self._evolving_model_mean)
        reference_model_radius = math.sqrt(self._reference_model_var) * self.m
        evolving_model_radius = math.sqrt(self._evolving_model_var) * self.m

        jaccard_index = 1.0
        if reference_model_radius + evolving_model_radius + drift_distance != 0.0:
            jaccard_index = self._jaccard_index(
                reference_model_radius, evolving_model_radius, drift_distance
            )

        if self._reference_model_n < 3.0 or self._evolving_model_n < 3.0:
            jaccard_index = 1.0

        if self._is_detection_enabled() and (jaccard_index < self.jaccard_threshold):
            _drift_detected = True

        self._drift_detected = _drift_detected

    def _is_detection_enabled(self):
        if (self._evolving_model_n < self.__min_samples) or (
            self._reference_model_n < self.__min_samples
        ):
            return False
        return True

    def _reference_model_teda_score(self, value):
        if self._reference_model_n < 3.0:
            return 1.0

        d = self._reference_model_mean - value
        if self._reference_model_var == 0.0:
            return 0.0

        norm_eccentricity = 1.0 / (2.0 * self._reference_model_n) + abs(d * d) / (
            2.0 * self._reference_model_n * self._reference_model_var
        )
        return norm_eccentricity

    def _reference_model_teda_threshold(self):
        if self._reference_model_n < 1.0:
            return 1.0
        return (self.m**2.0 + 1.0) / (2.0 * self._reference_model_n)

    def _update_reference_model(self, value):
        self._reference_model_n = self._reference_model_n + 1

        if self._reference_model_mean != value:
            model_weight = (self._reference_model_n - 1.0) / self._reference_model_n
            sample_weight = 1 - model_weight
            self._reference_model_mean = (model_weight * self._reference_model_mean) + (
                sample_weight * value
            )
            d = self._reference_model_mean - value
            self._reference_model_var = (model_weight * self._reference_model_var) + (
                sample_weight * abs(d * d)
            )

    def _update_evolving_model(self, value):
        if self.evolving_model_n < self.__min_samples:
            self._evolving_model_n += 1.0
        if self._evolving_model_mean != value:
            alpha_weight = (
                self.alpha
                if ((self._evolving_model_n - 1.0) / self._evolving_model_n) > self.alpha
                else ((self._evolving_model_n - 1.0) / self._evolving_model_n)
            )

            self._evolving_model_mean = (alpha_weight * self._evolving_model_mean) + (
                (1.0 - alpha_weight) * value
            )
            d = self._evolving_model_mean - value

            self._evolving_model_var = (alpha_weight * self._evolving_model_var) + (
                (1.0 - alpha_weight) * abs(d * d)
            )

    def _replace_reference_model(self):
        self._reference_model_n = min(self._evolving_model_n, self.__min_samples)
        self._reference_model_mean = self._evolving_model_mean
        self._reference_model_var = self._evolving_model_var

    def _jaccard_index(self, ra, rb, d):
        if ra + rb > d:
            inter = min(ra, rb - d) + d + min(rb, ra - d)
            union = max(ra, rb - d) + d + max(rb, ra - d)
            return inter / union
        return 0
