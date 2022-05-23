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
    ...     horizon=period
    ... )
    +1  SMAPE: 4.220171
    +2  SMAPE: 4.322648
    +3  SMAPE: 4.418546
    +4  SMAPE: 4.504986
    +5  SMAPE: 4.57924
    +6  SMAPE: 4.64123
    +7  SMAPE: 4.694042
    +8  SMAPE: 4.740753
    +9  SMAPE: 4.777291
    +10 SMAPE: 4.804558
    +11 SMAPE: 4.828114
    +12 SMAPE: 4.849823
    +13 SMAPE: 4.865871
    +14 SMAPE: 4.871972
    +15 SMAPE: 4.866274
    +16 SMAPE: 4.842614
    +17 SMAPE: 4.806214
    +18 SMAPE: 4.763355
    +19 SMAPE: 4.713455
    +20 SMAPE: 4.672062
    +21 SMAPE: 4.659102
    +22 SMAPE: 4.693496
    +23 SMAPE: 4.773707
    +24 SMAPE: 4.880654

    """

    def __init__(self, anomaly_detector, threshold: float):
        super().__init__(anomaly_detector, protect_anomaly_detector=False)
        self.threshold = threshold

    def classify(self, score):
        return score >= self.threshold
