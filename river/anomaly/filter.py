from river import anomaly


class ThresholdFilter(anomaly.base.AnomalyFilter):
    """Threshold anomaly filter.

    Parameters
    ----------
    anomaly_detector
        An anomaly detector.
    threshold
        A threshold above which to classify an anomaly score as anomalous.

    Examples
    --------

    Anomaly filters can be used as part of a pipeline. For instance, we might want to filter out
    anomalous observations so as not to corrupt a supervised model. As an example, let's take
    the `datasets.WaterFlow` dataset. Some of the samples have anomalous target variables because
    of human interventions. We don't want our model to learn these values.

    >>> from river import datasets
    >>> from river import metrics
    >>> from river import time_series

    >>> dataset = datasets.WaterFlow()
    >>> metric = metrics.SMAPE()

    >>> period = 24  # 24 samples per day

    >>> model = (
    ...     anomaly.ThresholdFilter(
    ...         anomaly.GaussianScorer(
    ...             window_size=period * 7,  # 7 days
    ...             grace_period=30
    ...         ),
    ...         threshold=0.995
    ...     ) |
    ...     time_series.HoltWinters(
    ...         alpha=0.3,
    ...         beta=0.1,
    ...         multiplicative=False
    ...     )
    ... )

    >>> time_series.evaluate(
    ...     dataset,
    ...     model,
    ...     metric,
    ...     horizon=period,
    ...     grace_period=period
    ... )
    +1  SMAPE: 4.171129
    +2  SMAPE: 4.247672
    +3  SMAPE: 4.310317
    +4  SMAPE: 4.358783
    +5  SMAPE: 4.391414
    +6  SMAPE: 4.414142
    +7  SMAPE: 4.430198
    +8  SMAPE: 4.440497
    +9  SMAPE: 4.443099
    +10 SMAPE: 4.439995
    +11 SMAPE: 4.432997
    +12 SMAPE: 4.424021
    +13 SMAPE: 4.412978
    +14 SMAPE: 4.396743
    +15 SMAPE: 4.374385
    +16 SMAPE: 4.343549
    +17 SMAPE: 4.301792
    +18 SMAPE: 4.256458
    +19 SMAPE: 4.213859
    +20 SMAPE: 4.180718
    +21 SMAPE: 4.167248
    +22 SMAPE: 4.1809
    +23 SMAPE: 4.2127
    +24 SMAPE: 4.263497

    """

    def __init__(self, anomaly_detector, threshold: float):
        super().__init__(anomaly_detector)
        self.threshold = threshold

    def classify(self, score):
        return score >= self.threshold
