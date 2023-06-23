from __future__ import annotations

import collections
import itertools
import random
import typing
import warnings

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
    seed
        Random seed for reproducibility.

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
    >>> import random
    >>> from river import drift

    >>> rng = random.Random(12345)
    >>> kswin = drift.KSWIN(alpha=0.0001, seed=42)

    >>> # Simulate a data stream composed by two data distributions
    >>> data_stream = rng.choices([0, 1], k=1000) + rng.choices(range(4, 8), k=1000)

    >>> # Update drift detector and verify if change is detected
    >>> for i, val in enumerate(data_stream):
    ...     _ = kswin.update(val)
    ...     if kswin.drift_detected:
    ...         print(f"Change detected at index {i}, input value: {val}")
    Change detected at index 1016, input value: 6

    References
    ----------
    [^1]: Christoph Raab, Moritz Heusinger, Frank-Michael Schleif, Reactive Soft Prototype Computing for
    Concept Drift Streams, Neurocomputing, 2020,

    """

    def __init__(
        self,
        alpha: float = 0.005,
        window_size: int = 100,
        stat_size: int = 30,
        seed: int | None = None,
        window: typing.Iterable | None = None,
    ):
        super().__init__()
        if alpha < 0 or alpha > 1:
            raise ValueError("Alpha must be between 0 and 1.")

        if window_size < 0:
            raise ValueError("window_size must be greater than 0.")

        if window_size < stat_size:
            raise ValueError("stat_size must be smaller than window_size.")

        self.alpha = alpha
        self.window_size = window_size
        self.stat_size = stat_size
        self.seed = seed

        self._reset()

        if window:
            self.window = collections.deque(window, maxlen=self.window_size)

    def _reset(self):
        super()._reset()
        self.p_value = 0
        self.n = 0
        self.window: collections.deque = collections.deque(maxlen=self.window_size)
        self._rng = random.Random(self.seed)

    def update(self, x):
        """Update the change detector with a single data point.

        Adds an element on top of the sliding window and removes the oldest one from the window.
        Afterwards, the KS-test is performed.

        Parameters
        ----------
        x
            New data sample the sliding window should add.

        Returns
        -------
        self

        """

        if self.drift_detected:
            self._reset()

        self.n += 1

        self.window.append(x)
        if len(self.window) >= self.window_size:
            rnd_window = [
                self.window[r]
                for r in self._rng.sample(range(self.window_size - self.stat_size), self.stat_size)
            ]
            most_recent = list(
                itertools.islice(self.window, self.window_size - self.stat_size, self.window_size)
            )

            # ks_2samp raises a warning when method="auto"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                st, self.p_value = stats.ks_2samp(rnd_window, most_recent, method="auto")

            if self.p_value <= self.alpha and st > 0.1:
                self._drift_detected = True
                self.window = collections.deque(most_recent, maxlen=self.window_size)
            else:
                self._drift_detected = False
        else:  # Not enough samples in the sliding window for a valid test
            self._drift_detected = False

        return self

    @classmethod
    def _unit_test_params(cls):
        yield {"seed": 1}
