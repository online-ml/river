from __future__ import annotations

import math

from river.base import DriftDetector


class TEDACDD(DriftDetector):
    r"""Concept Drift Detector based on Typically and Eccentrically Data Analytics (TEDA-CDD)

    Concept Drift Detector based on Typically and Eccentrically Data Analytics (TEDA-CDD) [^1] is a concept drift detector
    based on TEDA [^2], a framework for data analytic leveraging on typicality and eccentricity. It employs two models in
    monitoring the data stream in order to keep the information of a previous concept whereas monitoring the emergence
    of a new concept. The models are considered to represent to distinct concepts when the intersection of data samples
    are significantly low, described by the Jaccard Index (JI).

    TEDA-CDD has four essential components: reference data model, evolving data model, detection metric, and reset
    strategy. The reference model uses the classical definition of TEDA to represent the concept known by the classifier,
    disregarding any atypical data sample, whereas the evolving model uses an adaptation to describe the current features
    state.

    The model indicates a concept drift when the reference and evolving models are sufficiently distinct. In this context,
    a detection strategy based on the JI, considering the concept models as representations of sets, is proposed. Moreover,
    when a concept drift is detected, TEDA-CDD is reset to start monitoring the current concept; and, to avoid a cold
    restart, the information on the evolving model is used to update the reference model while a new evolving model is
    created from scratch.

    Parameters
    ----------
    m
        Sensitivity parameter [^2]. A higher value of `m` makes it difficult to encounter atypical data samples, whereas
        lower values causes more data samples to be considered atypical.
    alpha
        The forgetting factor, affecting the evolving model as a sensitivity parameter.
        The higher the value of `alpha`, the more important is a data sample to the current concept and the evolving
        model is more sensible to noise.
    jaccard_threshold
        The Jaccard Threshold (JT). A concept drift occurs if the jaccard index is lower than this given threshold.
        This threshold defines a limit of similarity between similar and dissimilar and the higher the value of JT, the
        more sensible to divergence is the CDD.

    Examples
    ----------
    >>> import random
    >>> from river import drift

    >>> rng = random.Random(12345)
    >>> teda_cdd = TEDACDD(m=1, alpha=0.95, jaccard_threshold=0.95)

    >>> # Simulate a data stream composed by two data distributions
    >>> data_stream = rng.choices([0, 1], k=1000) + rng.choices(range(4, 8), k=1000)

    >>> # Update drift detector and verify if change is detected
    >>> for i, val in enumerate(data_stream):
    ...     teda_cdd.update(val)

    References
    ----------
    [^1]: Y. T. P. Nunes, L. A. Guedes. 2024. Concept Drift Detection Based on Typicality and Eccentricity. IEEE Access
    12 (2024), pp. 13795-13808. doi: 10.1109/ACCESS.2024.3355959.
    [^2]: P. Angelov. 2014. Anomaly detection based on eccentricity analysis. 2014 IEEE Symposium on Evolving and
    Autonomous Learning Systems (EALS), Orlando, FL, USA, pp. 1-8. doi: 10.1109/EALS.2014.7009497.
    """

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
