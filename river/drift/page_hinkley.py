from river.base import DriftDetector


class PageHinkley(DriftDetector):
    """Page-Hinkley method for concept drift detection.

    This change detection method works by computing the observed
    values and their mean up to the current moment. Page-Hinkley
    does not signal warning zones, only change detections.
    The method works by means of the Page-Hinkley test. In general
    lines it will detect a concept drift if the observed mean at
    some instant is greater then a threshold value lambda.

    Parameters
    ----------
    min_instances
        The minimum number of instances before detecting change.
    delta
        The delta factor for the Page Hinkley test.
    threshold
        The change detection threshold (lambda).
    alpha
        The forgetting factor, used to weight the observed value and the mean.

    Examples
    --------
    >>> import numpy as np
    >>> from river.drift import PageHinkley
    >>> np.random.seed(12345)

    >>> ph = PageHinkley()

    >>> # Simulate a data stream composed by two data distributions
    >>> data_stream = np.concatenate((np.random.randint(2, size=1000),
    ...                               np.random.randint(4, high=8, size=1000)))

    >>> # Update drift detector and verify if change is detected
    >>> for i, val in enumerate(data_stream):
    ...     in_drift, in_warning = ph.update(val)
    ...     if in_drift:
    ...         print(f"Change detected at index {i}, input value: {val}")
    Change detected at index 1009, input value: 5

    References
    ----------
    [^1]: E. S. Page. 1954. Continuous Inspection Schemes. Biometrika 41, 1/2 (1954), 100–115.

    """

    def __init__(self, min_instances=30, delta=0.005, threshold=50, alpha=1 - 0.0001):
        super().__init__()
        self.min_instances = min_instances
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha
        self.x_mean = None
        self.sample_count = None
        self.sum = None
        self.reset()

    def reset(self):
        """Reset the change detector."""
        super().reset()
        self.sample_count = 1
        self.x_mean = 0.0
        self.sum = 0.0

    def update(self, value) -> tuple:
        """Update the change detector with a single data point.

        Parameters
        ----------
        value
            Input value

        Returns
        -------
        A tuple (drift, warning) where its elements indicate if a drift or a warning is detected.

        """
        if self._in_concept_change:
            self.reset()

        self.x_mean = self.x_mean + (value - self.x_mean) / float(self.sample_count)
        self.sum = max(0.0, self.alpha * self.sum + (value - self.x_mean - self.delta))

        self.sample_count += 1

        self._in_concept_change = False

        if self.sample_count < self.min_instances:
            return False, False

        if self.sum >= self.threshold:
            self._in_concept_change = True

        return self._in_concept_change, self._in_warning_zone
