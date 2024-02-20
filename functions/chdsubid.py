import typing
from collections import deque

import numpy as np


@typing.runtime_checkable
class SubIdentifier(typing.Protocol):
    def update(self, x: dict | np.ndarray, y: dict | np.ndarray):
        ...

    def eigs_modes(self) -> typing.Tuple[np.ndarray, np.ndarray]:
        ...


class SubIDDriftDetector:
    def __init__(
        self,
        subid: SubIdentifier,
        threshold: float = 0.1,
        train_size: int = 0,
        test_size: int = 10,
        time_lag: int = 0,
        grace_period: int = 0,
    ):
        self.subid = subid
        self.threshold = threshold
        if train_size == 0 and hasattr(subid, "window_size"):
            self.train_size = subid.window_size  # type: ignore
        else:
            self.train_size = train_size
        self.test_size = test_size
        self.time_lag = time_lag
        assert self.train_size > 0
        assert self.test_size > 0
        assert self.test_size + self.time_lag >= 0
        self.grace_period = grace_period
        self._drift_detected: bool
        self._Y = deque(maxlen=self.test_size + self.time_lag + self.test_size)

    def _compute_distance(self, Y: np.ndarray):
        _, W = self.subid.eigs_modes
        D = np.sum((Y @ Y.T) - (Y @ W @ W.T @ Y.T))
        return D

    def update(self, x, y):
        self.subid.update(x, y)
        self._Y.append(y)
        Y = np.array(self._Y)
        if Y.shape[0] > self.train_size + self.time_lag + self.grace_period:
            # TODO: Think about normalizing Ds w.r.t.
            #  (self.train_size * hankel_rank)? (Kawahara et al. 2007)
            D_train = (
                self._compute_distance(Y[: self.train_size, :])
                / self.train_size
            )
            # Must wait for all test samples to be collected
            # D_test = self._compute_distance(Y[-self.test_size :]) / self.test_size
            Y_test = Y[self.train_size + self.time_lag :, :]
            D_test = self._compute_distance(Y_test) / Y_test.shape[0]
            # TODO: Fix RuntimeWarning: invalid value encountered in scalar divide
            score = D_test / D_train
            print(score)
            self._drift_detected = abs(score) > self.threshold
        else:
            self._drift_detected = False
