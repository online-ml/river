from __future__ import annotations

from river import anomaly

import numpy as np


class KNNICAD(anomaly.base.SupervisedAnomalyDetector):
    """ KNN - Inductive Conformal Anomaly Detection (KNN-ICAD)

    - Map the time-series realization
    Examples
    --------

    >>> import numpy as np
    >>> from river import anomaly

    >>> np.random.seed(42)
    >>> Y = np.random.randn(150)

    >>> knn_cad = anomaly.KNNICAD(probationary_period=50, dim=10)

    >>> for y in Y:
    ...     knn_cad = knn_cad.learn_one(None, y)

    >>> knn_cad.score_one(None, Y[-1])
    0.1

    >>> knn_cad.score_one(None, -1)
    0.35

    >>> knn_cad.score_one(None, 0)
    0.075

    >>> knn_cad.score_one(None, 3)
    0.975

    """

    def __init__(self,
                 probationary_period: int = 10,
                 dim: int = 5):

        self.probationary_period = probationary_period
        self.dim = dim

        self.buffer = []
        self.training = []
        self.calibration = []
        self.scores = []
        self.sigma = np.diag(np.ones(self.dim))

        self.new_item = []
        self.new_score = []
        self.latest_anomaly_score = 0.0

        self.record_count = 0
        self.pred = -1
        self.k = 27

    def _metric(self, a, b):
        diff = a - np.array(b)

        return np.dot(np.dot(diff, self.sigma), diff.T)

    def _ncm(self, item, item_in_array=False):
        arr = [self._metric(x, item) for x in self.training]

        return np.sum(np.partition(arr, self.k + item_in_array)[:self.k + item_in_array])

    def handle_record(self, x, y):
        """
        Since the two functions `learn_one` and `predict_one` share a major part where a new instance is handled,
        the latest anomaly score is calculated, the buffer, training and scores are re-calculated, a common function,
        i.e `handle_record` is created.
        """

        self.buffer.append(y)
        self.record_count += 1

        if len(self.buffer) < self.dim:
            self.latest_anomaly_score = 0.0
            return
        else:
            self.new_item = self.buffer[-self.dim:]

            if self.record_count < self.probationary_period:
                self.training.append(self.new_item)
                self.latest_anomaly_score = 0.0
                return
            else:
                ost = self.record_count % self.probationary_period
                if ost == 0 or ost == int(self.probationary_period / 2):
                    try:
                        self.sigma = np.linalg.inv(
                            np.dot(np.array(self.training).T, self.training)
                        )
                    except np.linalg.linalg.LinAlgError:
                        print('Singular, non-invertible matrix at record', self.record_count)

                if len(self.scores) == 0:
                    self.scores = [self._ncm(v, True) for v in self.training]

                self.new_score = self._ncm(self.new_item)

                if self.record_count >= 2 * self.probationary_period:
                    self.training.pop(0)
                    self.training.append(self.calibration.pop(0))

                self.scores.pop(0)
                self.calibration.append(self.new_item)
                self.scores.append(self.new_score)

                if self.pred > 0:
                    self.pred -= 1
                    return 0.5
                elif self.latest_anomaly_score >= 0.9965:
                    self.pred = int(self.probationary_period / 5)

                self.latest_anomaly_score = 1. * len(np.where(np.array(self.scores) < self.new_score)[0]) / len(self.scores)

    def learn_one(self, x, y):

        self.handle_record(x, y)

        return self

    def score_one(self, x, y):

        # Checks whether score_one is used to score the instance that has just been seen by learn_one.
        # If yes, the part of handling record is skipped, since the latest anomaly score has already been computed.
        if y != self.buffer[-1]:
            self.handle_record(x, y)

        return self.latest_anomaly_score








