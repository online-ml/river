from __future__ import annotations

from river import anomaly, stats

__all__ = ["StandardAbsoluteDeviation"]


class StandardAbsoluteDeviation(anomaly.base.SupervisedAnomalyDetector):
    r"""Standard Absolute Deviation (SAD).

    SAD is the model that calculates the anomaly score by using the deviation from the mean/median, divided by the
    standard deviation of all the points seen within the data stream. The idea of this model is based on
    the $3 \times \sigma$ rule described in [^1].

    This implementation is adapted from the [implementation](https://github.com/selimfirat/pysad/blob/master/pysad/models/standard_absolute_deviation.py)
    within PySAD (Python Streaming Anomaly Detection) [^2].

    As a univariate anomaly detection algorithm, this implementation is adapted to `River` in a similar way as that of
    the `GaussianScorer` algorithm, with the variable taken into the account at the learning phase and scoring phase
    under variable `y`, ignoring `x`.

    Parameters
    ----------
    sub_stat
        The statistic to be subtracted, then divided by the standard deviation for scoring.
        This parameter must be either "mean" or "median".

    References
    ----------
    [^1]: Hochenbaum, J., Vallis, O.S., Kejariwal, A., 2017. Automatic Anomaly Detection in the Cloud Via
    Statistical Learning. https://doi.org/10.48550/ARXIV.1704.07706.
    [^2]: Yilmaz, S.F., Kozat, S.S., 2020. PySAD: A Streaming Anomaly Detection Framework in Python.
    https://doi.org/10.48550/ARXIV.2009.02572.

    Examples
    --------

    >>> import numpy as np
    >>> from river import anomaly
    >>> from river import stream

    >>> np.random.seed(42)

    >>> X = np.random.randn(150)

    >>> model = anomaly.StandardAbsoluteDeviation(sub_stat="mean")

    >>> for x in X:
    ...     model = model.learn_one(None, x)

    >>> model.score_one(None, 2)
    2.209735291993561

    >>> model.score_one(None, 0)
    0.08736408615569183

    >>> model.score_one(None, 1)
    1.1485496890746263

    """

    def __init__(self, sub_stat=None):
        self.variance = stats.Var()
        self.sub_stat = sub_stat

        if self.sub_stat == "mean":
            self.subtracted_statistic_estimator = stats.Mean()
        elif self.sub_stat == "median":
            self.subtracted_statistic_estimator = stats.Quantile(q=0.5)
        else:
            raise ValueError(
                f"Unknown subtracted statistic {self.sub_stat}, expected one of median, mean."
            )

    def learn_one(self, x, y):
        self.variance.update(y)
        self.subtracted_statistic_estimator.update(y)

        return self

    def score_one(self, x, y):
        score = (y - self.subtracted_statistic_estimator.get()) / (
            self.variance.get() ** 0.5 + 1e-10
        )

        return abs(score)
