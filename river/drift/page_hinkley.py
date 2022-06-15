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
    >>> import random
    >>> from river import drift

    >>> rng = random.Random(12345)
    >>> ph = drift.PageHinkley()

    >>> # Simulate a data stream composed by two data distributions
    >>> data_stream = rng.choices([0, 1], k=1000) + rng.choices(range(4, 8), k=1000)

    >>> # Update drift detector and verify if change is detected
    >>> for i, val in enumerate(data_stream):
    ...     _ = ph.update(val)
    ...     if ph.drift_detected:
    ...         print(f"Change detected at index {i}, input value: {val}")
    Change detected at index 1006, input value: 5

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

        self._reset()

    def _reset(self):
        super()._reset()
        self.sample_count = 1
        self.x_mean = 0.0
        self.sum = 0.0

    def update(self, x):
        if self._drift_detected:
            self._reset()

        self.x_mean += (x - self.x_mean) / self.sample_count
        self.sum = max(0.0, self.alpha * self.sum + (x - self.x_mean - self.delta))

        self.sample_count += 1

        self._drift_detected = False

        if self.sample_count >= self.min_instances:
            if self.sum >= self.threshold:
                self._drift_detected = True

        return self
