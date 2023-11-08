from __future__ import annotations

from river import anomaly

__all__ = ["ReconstructionAnomalyDetecion"]


class ReconstructionAnomalyDetecion(anomaly.base.AnomalyDetector):
    """Reconstruction Anomaly-Detection (RAD).
        This is the place for documentation
    """

    def __init__(self):
        print("Success")

    def learn_one(self, x):
        return self

    def score_one(self, x):
        return 1
