import numpy as np
from scipy import stats

from river.base import DriftDetector


class KSWIN(DriftDetector):
    r"""Kolmogorov-Smirnov Windowing method for concept drift detection.

    Parameters
    ----------
    alpha
        Probability for the test statistic of the Kolmogorov-Smirnov-Test. The alpha parameter is
        very sensitive, therefore should be set below 0.01.
    window_size
        Size of the sliding window.
    stat_size
        Size of the statistic window.
    window
        Already collected data to avoid cold start.

    Notes
    -----
    KSWIN (Kolmogorov-Smirnov Windowing) is a concept change detection method based
    on the Kolmogorov-Smirnov (KS) statistical test. KS-test is a statistical test with
    no assumption of underlying data distribution. KSWIN can monitor data or performance
    distributions. Note that the detector accepts one dimensional input as array.

    KSWIN maintains a sliding window $\Psi$ of fixed size $n$ (window_size). The
    last $r$ (stat_size) samples of $\Psi$ are assumed to represent the last
    concept considered as $R$. From the first $n-r$ samples of $\Psi$,
    $r$ samples are uniformly drawn, representing an approximated last concept $W$.

    The KS-test is performed on the windows $R$ and $W$ of the same size. KS
    -test compares the distance of the empirical cumulative data distribution $dist(R,W)$.

    A concept drift is detected by KSWIN if:

    $$
    dist(R,W) > \sqrt{-\frac{ln\alpha}{r}}
    $$

    The difference in empirical data distributions between the windows $R$ and $W$ is too large
    since $R$ and $W$ come from the same distribution.

    Examples
    --------
    >>> import numpy as np
    >>> from river.drift import KSWIN
    >>> np.random.seed(12345)

    >>> kswin = KSWIN()

    >>> # Simulate a data stream composed by two data distributions
    >>> data_stream = np.concatenate((np.random.randint(2, size=1000),
    ...                               np.random.randint(4, high=8, size=1000)))

    >>> # Update drift detector and verify if change is detected
    >>> for i, val in enumerate(data_stream):
    ...     in_drift, in_warning = kswin.update(val)
    ...     if in_drift:
    ...         print(f"Change detected at index {i}, input value: {val}")
    Change detected at index 1014, input value: 5
    Change detected at index 1828, input value: 6

    References
    ----------
    [^1]: Christoph Raab, Moritz Heusinger, Frank-Michael Schleif, Reactive Soft Prototype Computing for Concept Drift Streams, Neurocomputing, 2020,

    """

    def __init__(self, alpha=0.005, window_size=100, stat_size=30, window=None):
        super().__init__()
        self.window_size = window_size
        self.stat_size = stat_size
        self.alpha = alpha
        self.p_value = 0
        self.n = 0
        if self.alpha < 0 or self.alpha > 1:
            raise ValueError("Alpha must be between 0 and 1")

        if self.window_size < 0:
            raise ValueError("window_size must be greater than 0")

        if self.window_size < self.stat_size:
            raise ValueError("stat_size must be smaller than window_size")

        if not isinstance(window, np.ndarray) or window is None:
            self.window = np.array([])
        else:
            self.window = window

    def update(self, value) -> tuple:
        """Update the change detector with a single data point.

        Adds an element on top of the sliding window and removes the oldest one from the window.
        Afterwards, the KS-test is performed.

        Parameters
        ----------
        value
            New data sample the sliding window should add.

        Returns
        -------
        A tuple (drift, warning) where its elements indicate if a drift or a warning is detected.

        """
        self.n += 1
        currentLength = self.window.shape[0]
        if currentLength >= self.window_size:
            self.window = np.delete(self.window, 0)
            rnd_window = np.random.choice(self.window[: -self.stat_size], self.stat_size)

            (st, self.p_value) = stats.ks_2samp(
                rnd_window, self.window[-self.stat_size :], mode="exact"
            )

            if self.p_value <= self.alpha and st > 0.1:
                self._in_concept_change = True
                self.window = self.window[-self.stat_size :]
            else:
                self._in_concept_change = False
        else:  # Not enough samples in sliding window for a valid test
            self._in_concept_change = False

        self.window = np.concatenate([self.window, [value]])

        return self._in_concept_change, self._in_warning_zone

    def reset(self):
        """Reset the change detector."""
        self.p_value = 0
        self.window = np.array([])
        self._in_concept_change = False
