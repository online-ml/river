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

    def learn_one(self, x):
        self.anomaly_detector.learn_one(x)
        return self


class ConstantThresholder(Thresholder):
    """Constant thresholder.

    Each anomaly score is thresholded into a 0 or a 1.

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
               Precision   Recall   F1      Support
    <BLANKLINE>
           0       0.999    0.939   0.968      7975
           1       0.038    0.760   0.072        25
    <BLANKLINE>
       Macro       0.518    0.850   0.520
       Micro       0.939    0.939   0.939
    Weighted       0.996    0.939   0.965
    <BLANKLINE>
                     93.9% accuracy

    """

    def __init__(self, anomaly_detector: AnomalyDetector, threshold: float):
        super().__init__(anomaly_detector)
        self.threshold = threshold

    @classmethod
    def _unit_test_params(cls):
        from .hst import HalfSpaceTrees

        yield {"anomaly_detector": HalfSpaceTrees(), "threshold": 0.5}

    def score_one(self, x):
        return 1 if self.anomaly_detector.score_one(x) > self.threshold else 0


class QuantileThresholder(Thresholder):
    """Quantile thresholder.

    Each anomaly score is thresholded according to the value of the chosen quantile.

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
               Precision   Recall   F1      Support
    <BLANKLINE>
           0       0.998    0.983   0.991      7975
           1       0.087    0.520   0.149        25
    <BLANKLINE>
       Macro       0.543    0.751   0.570
       Micro       0.982    0.982   0.982
    Weighted       0.996    0.982   0.988
    <BLANKLINE>
                     98.2% accuracy

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

    def score_one(self, x):
        score = self.anomaly_detector.score_one(x)
        self.quantile.update(score)
        return 1 if score > self.quantile.get() else 0
