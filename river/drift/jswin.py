from __future__ import annotations

import collections
import itertools
import random
import typing

import numpy as np

from river.base import DriftDetector

EPS = 1e-8


class JSWIN(DriftDetector):
    r"""Jensen-Shannon Windowing method for concept drift detection.

    Parameters
    ----------
    alpha
        Probability for the test statistic of the Jensen-Shannon-Test. The alpha parameter is
        very sensitive, therefore should be set below 0.01.
    window_size
        Size of the sliding window (must be even).
    bin_size
        Size of the bin.
    seed
        Random seed for reproducibility.
    window
        Already collected data to avoid cold start.
    Notes
    -----
    JSWIN (Jensen-Shannon Windowing) is a concept change detection method based
    on the Jensen-Shannon (JS) divergence. JS divergence is a method of measuring the
    similarity between two probability distributions, but unlike the Kullback-Leibner
    divergence is symmetric and has upper bound limit, in fact its values are from range [0,1].
    JSWIN can monitor data or performance distributions. Note that the detector accepts one dimensional
    input as array. JSWIN maintains a sliding windows $\Psi$ of fixed size $n$ (window_size).
    The last $n$ samples of $\Psi$ are assumed to represent the last concept considered as $Q$.
    The first $n$ samples of $\Psi$ represent the prevoius concept $P$.

    The Jensen-Shannon divergence is performed on the windows $P$ and $Q$ of the same size.
    A concept drift is detected by JSWIN if:

    $$
    JS(P,Q) = \frac{1}{2} \left( D_{KL}(P||M) + D_{KL}(Q||M) \right) > \alpha
    $$

    where $D_{KL}$ is the Kullback-Leibler divergence and $M$ is the average of the two distributions:
    $$

    Examples
    --------
    >>> import random
    >>> from river import drift

    >>> rng = random.Random(12345)
    >>> jswin = drift.JSWIN(alpha=0.6)

    >>> # Simulate a data stream composed by two data distributions
    >>> data_stream = rng.choices([0, 1], k=1000) + rng.choices(range(4, 8), k=1000)

    >>> # Update drift detector and verify if change is detected
    >>> for i, val in enumerate(data_stream):
    ...     jswin.update(val)
    ...     if jswin.drift_detected:
    ...         print(f"Change detected at index {i}, input value: {val}")
    Change detected at index 1026, input value: 7
    """

    def __init__(
        self,
        alpha: float = 0.1,
        window_size: int = 100,
        bin_size: int = 10,
        seed: int | None = None,
        window: typing.Iterable | None = None,
    ):
        super().__init__()
        if alpha < 0 or alpha > 1:
            raise ValueError("Alpha must be between 0 and 1.")

        if window_size < 0:
            raise ValueError("window_size must be greater than 0.")

        if window_size % 2 != 0:
            raise ValueError("window_size must be even.")

        if bin_size < 0:
            raise ValueError("bin_size must be greater than 0.")

        if window_size // bin_size < 4:
            raise ValueError("window_size must be greater than 4 * bin_size.")

        self.alpha = alpha
        self.window_size = window_size
        self.bin_size = bin_size
        self.seed = seed
        self._js_value: float | None = None

        self._reset()

        if window:
            self.window = collections.deque(window, maxlen=self.window_size)
        else:
            self.window = collections.deque(maxlen=self.window_size)

    def _reset(self):
        super()._reset()
        self._rng = random.Random(self.seed)

    def update(self, x: float) -> None:
        """Update the detector with a new sample.

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

        self.window.append(x)
        if len(self.window) >= self.window_size:
            first_window = list(itertools.islice(self.window, self.window_size // 2))
            second_window = list(
                itertools.islice(self.window, self.window_size // 2, self.window_size)
            )

            self._js_value = self._jensen_shannon_divergence(first_window, second_window)

            if self._js_value > self.alpha:
                self._drift_detected = True
                self.window = collections.deque(second_window, maxlen=self.window_size)
            else:
                self._drift_detected = False
        else:
            self._drift_detected = False

    def _kullback_leibler_divergence(self, p: list[float], q: list[float]) -> float:
        """Calculate the Kullback-Leibler divergence between two distributions.
        Parameters
        ----------
        p
            First distribution.
        q
            Second distribution.
        Returns
        -------
        float
            Kullback-Leibler divergence.
        """
        points = sorted(set(p + q))
        bins = [points[i] for i in range(0, len(points), max(1, len(points) // self.bin_size))]
        p_hist, _ = np.histogram(p, bins=bins, density=True)
        q_hist, _ = np.histogram(q, bins=bins, density=True)
        p_hist = p_hist / np.sum(p_hist)
        q_hist = q_hist / np.sum(q_hist)
        return np.sum(
            np.where(
                (p_hist != 0) & (q_hist != 0),
                p_hist * np.log(p_hist / (q_hist + EPS)) - p_hist + q_hist,
                0,
            )
        )

    def _jensen_shannon_divergence(self, p: list[float], q: list[float]) -> float:
        """Calculate the Jensen-Shannon divergence between two distributions.
        Parameters
        ----------
        p
            First distribution.
        q
            Second distribution.
        Returns
        -------
        float
            Jensen-Shannon divergence.
        """
        m = [(p[i] + q[i]) / 2 for i in range(len(p))]
        return 0.5 * (
            self._kullback_leibler_divergence(p, m) + self._kullback_leibler_divergence(q, m)
        )
