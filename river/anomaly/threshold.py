from river.base import WrapperMixin

from .base import AnomalyDetector


__all__ = ["ConstantThresholder"]


class Thresholder(AnomalyDetector, WrapperMixin):
    def __init__(self, anomaly_detector: AnomalyDetector):
        self.anomaly_detector = anomaly_detector

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
       Macro       0.519    0.850   0.520
       Micro       0.939    0.939   0.939
    Weighted       0.996    0.939   0.966
    <BLANKLINE>
                     93.9% accuracy

    """

    def __init__(self, anomaly_detector: AnomalyDetector, threshold: float):
        super().__init__(anomaly_detector)
        self.threshold = threshold

    def score_one(self, x):
        return 1 if self.anomaly_detector.score_one(x) > self.threshold else 0
