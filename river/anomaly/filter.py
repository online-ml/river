from river import anomaly


class ThresholdFilter(anomaly.base.AnomalyFilter):
    def __init__(self, anomaly_detector, threshold: float):
        super().__init__(anomaly_detector)
        self.threshold = threshold

    def classify(self, score):
        return score >= self.threshold
