import typing
from collections import deque

import numpy as np


@typing.runtime_checkable
class SubIdentifier(typing.Protocol):
    def update(self, x: dict | np.ndarray, y: dict | np.ndarray):
        ...

    @property
    def eig(self) -> typing.Tuple[np.ndarray, np.ndarray]:
        ...


class SubIDDriftDetector:
    def __init__(
        self,
        subid: SubIdentifier,
        ref_size: int,
        test_size: int,
        threshold: float = 0.1,
        time_lag: int = 0,
        grace_period: int = 0,
    ):
        self.subid = subid
        self.threshold = threshold
        if ref_size == 0 and hasattr(subid, "window_size"):
            self.ref_size = subid.window_size  # type: ignore
        else:
            self.ref_size = ref_size
        self.test_size = test_size
        self.time_lag = time_lag
        self.grace_period = grace_period
        assert self.ref_size > 0
        assert self.test_size > 0
        assert self.test_size + self.time_lag >= 0
        assert self.grace_period < self.test_size
        self.grace_period = grace_period
        self.drift_detected: bool
        self.score: float
        self._Y = deque(maxlen=self.ref_size + self.time_lag + self.test_size)

    def _compute_distance(self, Y: np.ndarray) -> float:
        """Compute the distance between the Hankel matrix and its transformation.

        This formulation computes a measure of how much information in the dataset represented by Y is preserved or retained when projected onto the space spanned by W. The difference between the covariance matrix of Y and the projected version is computed, and the sum of all elements in this difference matrix gives an overall measure of dissimilarity or distortion.

        Args:
            Y): Hankel matrix

        Returns:
            Distance between the Hankel matrix and its transformation.
        """
        _, W = self.subid.eig
        W = W.real
        D = np.sum((Y @ Y.T) - (Y @ W @ W.T @ Y.T))
        return D

    def update(self, x, y):
        self.subid.update(x, y)

        self._Y.append(y)
        Y = np.array(self._Y)
        if Y.shape[0] > self.ref_size + self.time_lag + self.grace_period:
            # TODO: Think about normalizing Ds w.r.t.
            #  (self.ref_size * hankel_rank)? (Kawahara et al. 2007)
            D_train = (
                self._compute_distance(Y[: self.ref_size, :]) / self.ref_size
            )
            # Must wait for all test samples to be collected
            # D_test = self._compute_distance(Y[-self.test_size :]) / self.test_size
            Y_test = Y[self.ref_size + self.time_lag :, :]
            D_test = self._compute_distance(Y_test) / Y_test.shape[0]
            # TODO: Fix RuntimeWarning: invalid value encountered in scalar divide
            # TODO: Learn why always positive in Kawahara et al. (2007)
            self.score = abs(D_test / D_train)
            self.drift_detected = self.score > self.threshold
        else:
            self.score = 0.0
            self.drift_detected = False
