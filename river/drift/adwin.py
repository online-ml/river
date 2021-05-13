from river.base import DriftDetector

from .adwin_c import AdaptiveWindowing


class ADWIN(DriftDetector):
    r"""Adaptive Windowing method for concept drift detection.

    ADWIN (ADaptive WINdowing) is a popular drift detection method with
    mathematical guarantees. ADWIN efficiently keeps a variable-length window
    of recent items; such that it holds that there has no been change in the
    data distribution. This window is further divided into two sub-windows
    $(W_0, W_1)$ used to determine if a change has happened. ADWIN compares
    the average of $W_0$ and $W_1$ to confirm that they correspond to the
    same distribution. Concept drift is detected if the distribution equality
    no longer holds. Upon detecting a drift, $W_0$ is replaced by $W_1$ and a
    new $W_1$ is initialized. ADWIN uses a confidence value
    $\delta=\in(0,1)$ to determine if the two sub-windows correspond to the
    same distribution.

    **Input**: `value` can be any numeric value related to the definition of
    concept change for the data analyzed. For example, using 0's or 1's
    to track drift in a classifier's performance as follows:

    - 0: Means the learners prediction was wrong

    - 1: Means the learners prediction was correct

    Parameters
    ----------
    delta
        Confidence value.

    Examples
    --------
    >>> import numpy as np
    >>> from river.drift import ADWIN
    >>> np.random.seed(12345)

    >>> adwin = ADWIN()

    >>> # Simulate a data stream composed by two data distributions
    >>> data_stream = np.concatenate((np.random.randint(2, size=1000),
    ...                               np.random.randint(4, high=8, size=1000)))

    >>> # Update drift detector and verify if change is detected
    >>> for i, val in enumerate(data_stream):
    ...     in_drift, in_warning = adwin.update(val)
    ...     if in_drift:
    ...         print(f"Change detected at index {i}, input value: {val}")
    Change detected at index 1023, input value: 5
    Change detected at index 1055, input value: 7
    Change detected at index 1087, input value: 5
    Change detected at index 1119, input value: 7

    References
    ----------
    [^1]: Albert Bifet and Ricard Gavalda.
    "Learning from time-changing data with adaptive windowing."
    In Proceedings of the 2007 SIAM international conference on data mining,
    pp. 443-448. Society for Industrial and Applied Mathematics, 2007.

    """

    def __init__(self, delta=0.002):
        super().__init__()
        self._helper = AdaptiveWindowing(delta)

    @property
    def delta(self):
        return self._helper.get_delta()

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
    def estimation(self):
        """Error estimation"""
        if self.width == 0:
            return 0.0
        return self.total / self.width

    def update(self, value):
        """Update the change detector with a single data point.

        Apart from adding the element value to the window, by inserting it in
        the correct bucket, it will also update the relevant statistics, in
        this case the total sum of all values, the window width and the total
        variance.

        Parameters
        ----------
        value
            Input value

        Returns
        -------
        tuple
            A tuple (drift, warning) where its elements indicate if a drift or a warning is
            detected.

        """
        self._in_concept_change = self._helper.update(value)
        return self._in_concept_change, self._in_warning_zone

    def reset(self):
        """Reset the change detector."""
        self._helper = AdaptiveWindowing(delta=self.delta)
