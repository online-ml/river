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
        Defaults to `stats.Mean()`.

    References
    ----------
    [^1]: Hochenbaum, J., Vallis, O.S., Kejariwal, A., 2017. Automatic Anomaly Detection in the Cloud Via
    Statistical Learning. https://doi.org/10.48550/ARXIV.1704.07706.
    [^2]: Yilmaz, S.F., Kozat, S.S., 2020. PySAD: A Streaming Anomaly Detection Framework in Python.
    https://doi.org/10.48550/ARXIV.2009.02572.

    Examples
    --------

    >>> import random
    >>> from river import anomaly
    >>> from river import stats
    >>> from river import stream

    >>> rng = random.Random(42)

    >>> model = anomaly.StandardAbsoluteDeviation(sub_stat=stats.Mean())

    >>> for _ in range(150):
    ...     y = rng.gauss(0, 1)
    ...     model.learn_one(None, y)

    >>> model.score_one(None, 2)
    2.057...

    >>> model.score_one(None, 0)
    0.084...

    >>> model.score_one(None, 1)
    0.986...

    """

    def __init__(self, sub_stat: stats.base.Univariate | None = None):
        self.variance = stats.Var()
        self.sub_stat = sub_stat or stats.Mean()

    def learn_one(self, x, y):
        self.variance.update(y)
        self.sub_stat.update(y)

    def score_one(self, x, y):
        score = (y - self.sub_stat.get()) / (self.variance.get() ** 0.5 + 1e-10)

        return abs(score)
