from river.base import Wrapper
from river.stats import Quantile

from .base import AnomalyDetector

__all__ = ["ConstantThresholder", "QuantileThresholder"]


class Thresholder(AnomalyDetector, Wrapper):
    def __init__(self, anomaly_detector: AnomalyDetector):
        self.anomaly_detector = anomaly_detector

    @property
    def _wrapped_model(self):
        return self.anomaly_detector

    def learn_one(self, *args):
        self.anomaly_detector.learn_one(*args)
        return self


class ConstantThresholder(Thresholder):
    """Constant thresholder.

    A thresholder is a wrapper around an existing anomaly detector. It converts the latter's
    anomaly score into a boolean. This allows turning the anomaly detector into a binary
    classifier.

    This thresholder converts the score into a `True` if the score is above a specified threshold,
    and else converts the score into a `False`.

    Parameters
    ----------
    anomaly_detector
        The anomaly detector that will learn and produce scores.
    threshold
        The threshold to apply.

    Examples
    --------

    >>> from river import anomaly
    >>> from river import compose
    >>> from river import datasets
    >>> from river import metrics
    >>> from river import preprocessing

    >>> model = compose.Pipeline(
    ...     preprocessing.MinMaxScaler(),
    ...     anomaly.ConstantThresholder(
    ...         anomaly.HalfSpaceTrees(seed=42),
    ...         threshold=0.8
    ...     )
    ... )

    >>> report = metrics.ClassificationReport()

    >>> for x, y in datasets.CreditCard().take(8000):
    ...     score = model.score_one(x)
    ...     model = model.learn_one(x)
    ...     report = report.update(y, score)

    >>> report
               Precision   Recall   F1       Support
    <BLANKLINE>
           0      99.92%   93.92%   96.83%      7975
           1       3.77%   76.00%    7.18%        25
    <BLANKLINE>
       Macro      51.84%   84.96%   52.00%
       Micro      93.86%   93.86%   93.86%
    Weighted      99.62%   93.86%   96.55%
    <BLANKLINE>
                     93.86% accuracy

    """

    def __init__(self, anomaly_detector: AnomalyDetector, threshold: float):
        super().__init__(anomaly_detector)
        self.threshold = threshold

    @classmethod
    def _unit_test_params(cls):
        from .hst import HalfSpaceTrees

        yield {"anomaly_detector": HalfSpaceTrees(), "threshold": 0.5}

    def score_one(self, *args):
        return self.anomaly_detector.score_one(*args) > self.threshold


class QuantileThresholder(Thresholder):
    """Quantile thresholder.

    A thresholder is a wrapper around an existing anomaly detector. It converts the latter's
    anomaly score into a boolean. This allows turning the anomaly detector into a binary
    classifier.

    This thresholder converts the score into a `True` if the score is above a specified quantile,
    and else converts the score into a `False`.

    Parameters
    ----------
    anomaly_detector
        The anomaly detector that will learn and produce scores.
    q
        Determines the quantile to compute. Should be comprised between 0 and 1.

    Examples
    --------

    >>> from river import anomaly
    >>> from river import compose
    >>> from river import datasets
    >>> from river import metrics
    >>> from river import preprocessing

    >>> model = compose.Pipeline(
    ...     preprocessing.MinMaxScaler(),
    ...     anomaly.QuantileThresholder(
    ...         anomaly.HalfSpaceTrees(seed=42),
    ...         q=0.95
    ...     )
    ... )

    >>> report = metrics.ClassificationReport()

    >>> for x, y in datasets.CreditCard().take(8000):
    ...     score = model.score_one(x)
    ...     model = model.learn_one(x)
    ...     report = report.update(y, score)

    >>> report
               Precision   Recall   F1       Support
    <BLANKLINE>
           0      99.85%   98.29%   99.06%      7975
           1       8.72%   52.00%   14.94%        25
    <BLANKLINE>
       Macro      54.29%   75.15%   57.00%
       Micro      98.15%   98.15%   98.15%
    Weighted      99.56%   98.15%   98.80%
    <BLANKLINE>
                     98.15% accuracy

    """

    def __init__(self, anomaly_detector: AnomalyDetector, q: float):
        super().__init__(anomaly_detector)
        self.quantile = Quantile(q=q)

    @classmethod
    def _unit_test_params(cls):
        from .hst import HalfSpaceTrees

        yield {"anomaly_detector": HalfSpaceTrees(), "q": 0.5}

    @property
    def q(self):
        return self.quantile.q

    def score_one(self, *args):
        score = self.anomaly_detector.score_one(*args)
        self.quantile.update(score)
        return score > self.quantile.get()
