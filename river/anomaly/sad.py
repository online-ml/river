from __future__ import annotations

from river import anomaly, stats

__all__ = ["StandardAbsoluteDeviation"]


class StandardAbsoluteDeviation(anomaly.base.AnomalyDetector):
    r"""Standard Absolute Deviation (SAD).

    SAD is the model that calculates the anomaly score by using the deviation from the mean/median, divided by the
    standard deviation of all the points seen within the data stream. The idea of this model is based on
    the $3 \times \sigma$ rule described in [^1].

    This implementation is adapted from the [implementation](https://github.com/selimfirat/pysad/blob/master/pysad/models/standard_absolute_deviation.py)
    within PySAD (Python Streaming Anomaly Detection) [^2].

    Despite the fact that this model only works with univariate distribution, the author maintains the required input
    to be a dictionary (with length 1) to align with other anomaly detection algorithms implemented within `River`.

    Parameters
    ----------
    sub_stat
        The statistic to be substracted, then divided by the standard deviation for scoring.
        This parameter must be either "mean" or "median".
    kwargs
        Other parameters passed to variance attribute, particularly the delta degree of freedom (`ddof`).

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

    >>> X = np.random.randn(150, 1)

    >>> model = anomaly.StandardAbsoluteDeviation(sub_stat="mean")

    >>> for x, _ in stream.iter_array(X):
    ...     model.learn_one(x)

    >>> model.score_one({0: 2})
    2.209735291993561

    >>> model.score_one({0: 0})
    0.08736408615569183

    >>> model.score_one({0: -1})
    0.9738215167632427

    """

    def __init__(self, sub_stat="mean"):
        if sub_stat == "mean":
            self.subtracted_statistic = stats.Mean()
        elif sub_stat == "median":
            self.subtracted_statistic = stats.Quantile(q=0.5)
        else:
            raise ValueError(
                f"Unknown subtracted statistic {sub_stat}, expected one of median, mean."
            )

        self.variance = stats.Var()

    def learn_one(self, x):
        assert len(x) == 1
        ((x_key, x_value),) = x.items()

        self.variance.update(x_value)
        self.subtracted_statistic.update(x_value)

    def score_one(self, x):
        assert len(x) == 1
        ((x_key, x_value),) = x.items()

        score = (x_value - self.subtracted_statistic.get()) / (self.variance.get() ** 0.5 + 1e-10)

        return abs(score)
