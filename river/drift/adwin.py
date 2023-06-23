from __future__ import annotations

from river.base import DriftDetector

from .adwin_c import AdaptiveWindowing


class ADWIN(DriftDetector):
    r"""Adaptive Windowing method for concept drift detection.

    ADWIN (ADaptive WINdowing) is a popular drift detection method with mathematical guarantees.
    ADWIN efficiently keeps a variable-length window of recent items; such that it holds that
    there has no been change in the data distribution. This window is further divided into two
    sub-windows $(W_0, W_1)$ used to determine if a change has happened. ADWIN compares the average of
    $W_0$ and $W_1$ to confirm that they correspond to the same distribution. Concept drift is detected
    if the distribution equality no longer holds. Upon detecting a drift, $W_0$ is replaced by $W_1$ and a
    new $W_1$ is initialized. ADWIN uses a significance value $\delta=\in(0,1)$ to determine
    if the two sub-windows correspond to the same distribution.

    Parameters
    ----------
    delta
        Significance value.
    clock
        How often ADWIN should check for change. 1 means every new data point, default is 32.
        Higher values speed up processing, but may also lead to increased delay in change
        detection.
    max_buckets
        The maximum number of buckets of each size that ADWIN should keep before merging buckets
        (default is 5).
    min_window_length
        The minimum length of each subwindow (default is 5). Lower values may decrease delay in
        change detection but may also lead to more false positives.
    grace_period
        ADWIN does not perform any change detection until at least this many data points have
        arrived (default is 10).

    Examples
    --------
    >>> import random
    >>> from river import drift

    >>> rng = random.Random(12345)
    >>> adwin = drift.ADWIN()

    >>> # Simulate a data stream composed by two data distributions
    >>> data_stream = rng.choices([0, 1], k=1000) + rng.choices(range(4, 8), k=1000)

    >>> # Update drift detector and verify if change is detected
    >>> for i, val in enumerate(data_stream):
    ...     _ = adwin.update(val)
    ...     if adwin.drift_detected:
    ...         print(f"Change detected at index {i}, input value: {val}")
    Change detected at index 1023, input value: 4

    References
    ----------
    [^1]: Albert Bifet and Ricard Gavalda.
    "Learning from time-changing data with adaptive windowing."
    In Proceedings of the 2007 SIAM international conference on data mining,
    pp. 443-448. Society for Industrial and Applied Mathematics, 2007.

    """

    def __init__(self, delta=0.002, clock=32, max_buckets=5, min_window_length=5, grace_period=10):
        super().__init__()
        self.delta = delta
        self.clock = clock
        self.max_buckets = max_buckets
        self.min_window_length = min_window_length
        self.grace_period = grace_period
        self._reset()

    def _reset(self):
        super()._reset()
        self._helper = AdaptiveWindowing(
            delta=self.delta,
            clock=self.clock,
            max_buckets=self.max_buckets,
            min_window_length=self.min_window_length,
            grace_period=self.grace_period,
        )

    @property
    def width(self):
        """Window size"""
        return self._helper.get_width()

    @property
    def n_detections(self):
        return self._helper.get_n_detections()

    @property
    def variance(self):
        return self._helper.get_variance()

    @property
    def total(self):
        return self._helper.get_total()

    @property
    def estimation(self) -> float:
        """Estimate of mean value in the window."""
        if self.width == 0:
            return 0.0
        return self.total / self.width

    def update(self, x):
        """Update the change detector with a single data point.

        Apart from adding the element value to the window, by inserting it in
        the correct bucket, it will also update the relevant statistics, in
        this case the total sum of all values, the window width and the total
        variance.

        Parameters
        ----------
        x
            Input value

        Returns
        -------
        self

        """
        if self.drift_detected:
            self._reset()

        self._drift_detected = self._helper.update(x)

        return self
