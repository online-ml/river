from __future__ import annotations

from typing import Union, List

import copy

from river import anomaly

import numpy as np


class KNNInductiveCAD(anomaly.base.SupervisedAnomalyDetector):
    r"""KNN - Inductive Conformal Anomaly Detection (KNN-ICAD)

    KNN-ICAD [^1] is a conformalized density- and distance-based anoaly detection algorithms for a
    one-dimensional time-series data. This algorithm uses a combination of a feature extraction method, an approach to
    assess a score whether a new observation differs significantly from a previously observed data, and a probabilistic
    interpretation of this score based on the conformal paradigm.

    The basic idea of conformal prediction is that: based on a training set $(x_1, x_2, ..., x_m)$ for a new observation
    $x_{m+1}$, the parameter $p$ is computed. $p$ is defined as the probability that in the training set an observation
    with a more extreme value of a Non-Conformity Measure (NCM) than that of the new observation can be found. CP method
    is usually simple to be adjusted to anomaly detection problems, as Conformal Anomaly Detection - CAD.

    However, due to a high level of complexity, this algorithm uses a modification of CAD - Inductive CAD (or ICAD). The
    main idea of ICAD consists of two datasets - "proper training" and "calibration" sets. For each data point from the
    calibration set. The NCM value is calculated using the proper training set, moroveor, each newly observed data point
    will also have its NCM value calculated. The parameter $p$ will then be estimated by comparing the sets of NCM
    values.

    As a univariate anomaly detection algorithm, this implementation is adapted to `River` in a similar way as that of
    the `GaussianScorer` or `StandardAbsoluteDeviation` algorithms, with the variable taken into the account at the
    learning phase and scoring phase under variable `y`, ignoring `x`.

    This implementation is adapted from the implementation within
    [PySAD](https://github.com/selimfirat/pysad/blob/master/pysad/models/knn_cad.py)
    (Python Streaming Anomaly Detection) and
    [NAB](https://github.com/numenta/NAB/blob/master/nab/detectors/knncad/knncad_detector.py)
    (Numenta Anomaly Benchmark).

    Parameters
    ----------
    probationary_period
        The probationary period of the model. Within this period, the anomaly detectior is allowed to learn the
        data patterns without being tested.
        The input of this parameter must be of type integer.
    dim
        The length of the buffer, which corresponds to the vertical dimension of the training and calibration set.
        The input of this parameter must be of type integer.
    partition_position
        The position for the array to be partitioned within the function to calculate the NCM.
        This parameter will be passed through the `np.partition` command within the previously siad function.
        The input of this parameter must be of type integer.

    References
    ----------
    [^1]: Burnaev, E. and Ishimtsev, V. (2016) 'Conformalized density- and distance-based anomaly detection
    in time-series data,' arXiv [Preprint]. https://doi.org/10.48550/arxiv.1608.04585.
    [^2]: Glenn Shafer and Vladimir Vovk. A tutorial on conformal prediction. Journal of Machine Learning Research,
    9(Mar):371–421, 2008.
    [^3]: Rikard Laxhammar and Goran Falkman. Inductive conformal anomaly detection for sequential detection of
    anomalous sub-trajectories. Annals of Mathematics and Artificial Intelligence, 74(1-2):67–94, 2015.

    Examples
    --------

    >>> import numpy as np
    >>> from river import anomaly

    >>> np.random.seed(42)
    >>> stream = np.random.randn(1000)

    >>> knn_cad = anomaly.KNNInductiveCAD(probationary_period=100, dim=20)

    >>> for elem in stream:
    ...     knn_cad = knn_cad.learn_one(None, elem)

    >>> knn_cad.score_one(None, stream[-1])
    0.2375

    # When the algorithm scores an observation that has already been seen, only the latest anomaly score is queried.
    # No actual update on the model is conducted.
    >>> knn_cad.record_count == 1000
    True

    >>> knn_cad.score_one(None, -1)
    0.25

    >>> knn_cad.score_one(None, 0)
    0.1875

    >>> knn_cad.score_one(None, 3)
    0.4375

    """

    def __init__(self, probationary_period: int = 10, dim: int = 5, partition_position: int = 27):
        self.probationary_period = probationary_period
        self.dim = dim
        self.partition_position = partition_position

        self.buffer: List[Union[int, float]] = []
        self.training: List[Union[int, float]] = []
        self.calibration: List[Union[int, float]] = []
        self.scores: List[Union[int, float]] = []
        self.sigma = np.diag(np.ones(self.dim))

        self.new_score: List[Union[int, float]] = []
        self.latest_anomaly_score = 0.0

        self.record_count = 0
        self.pred = -1

    def _metric(self, a, b):
        diff = a - np.array(b)

        return np.dot(np.dot(diff, self.sigma), diff.T)

    def _non_conformity_measure(self, item, item_in_array=False):
        arr = [self._metric(x, item) for x in self.training]

        return np.sum(
            np.partition(arr, kth=(self.partition_position + item_in_array))[
                : (self.partition_position + item_in_array)
            ]
        )

    def handle_record(self, x, y):
        """
        Since the two functions `learn_one` and `predict_one` share a major part where a new instance is handled,
        the latest anomaly score is calculated, the buffer, training and scores are re-calculated, a common function,
        i.e `handle_record` is created.

        This function only takes `y` into account, however, `x` is included as an argument for consistency.
        """

        self.buffer.append(y)
        self.record_count += 1

        if len(self.buffer) < self.dim:
            self.latest_anomaly_score = 0.0
            return
        else:
            # Remove one element from the buffer as a new element has already been appended to make sure that
            # the number of elements within this buffer will always be equal to the dimension.
            if len(self.buffer) != self.dim:
                self.buffer.pop(0)

            if self.record_count < self.probationary_period:
                self.training.append(copy.deepcopy(self.buffer))
                self.latest_anomaly_score = 0.0
                return
            else:
                ost = self.record_count % self.probationary_period
                if ost == 0 or ost == int(self.probationary_period / 2):
                    try:
                        self.sigma = np.linalg.inv(np.dot(np.array(self.training).T, self.training))
                    except np.linalg.linalg.LinAlgError:
                        print(f"Singular, non-invertible matrix at record {self.record_count}.")

                if len(self.scores) == 0:
                    self.scores = [self._non_conformity_measure(v, True) for v in self.training]

                self.new_score = self._non_conformity_measure(copy.deepcopy(self.buffer))

                if self.record_count >= 2 * self.probationary_period:
                    self.training.pop(0)
                    self.training.append(self.calibration.pop(0))

                self.scores.pop(0)
                self.calibration.append(copy.deepcopy(self.buffer))
                self.scores.append(self.new_score)

                if self.pred > 0:
                    self.pred -= 1
                    return 0.5
                elif self.latest_anomaly_score >= 0.9965:
                    self.pred = int(self.probationary_period / 5)

                self.latest_anomaly_score = (
                    1.0
                    * len(np.where(np.array(self.scores) < self.new_score)[0])
                    / len(self.scores)
                )

    def learn_one(self, x, y):
        self.handle_record(x, y)

        return self

    def score_one(self, x, y):
        """
        Checks whether score_one is used to score the instance that has just been seen by learn_one.
        If yes, the part of handling record is skipped, since the latest anomaly score has already been computed.
        Else, the record is handled normally.
        """
        if y != self.buffer[-1]:
            self.handle_record(x, y)

        return self.latest_anomaly_score
