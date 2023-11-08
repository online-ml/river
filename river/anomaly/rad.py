from __future__ import annotations

from river import anomaly, base

__all__ = ["ReconstructionAnomalyDetecion"]


class ReconstructionAnomalyDetecion(anomaly.base.SupervisedAnomalyDetector):
    """Reconstruction Anomaly-Detection (RAD).
    This is the place for documentation
    """

    def __init__(self):
        print("Success")

    def learn_one(self, x: dict, y: base.typing.Target):
        """Update the model.

        Parameters
        ----------
        x
            A dictionary of features.

        Returns
        -------
        self

        """
        return self

    def score_one(self, x: dict, y: base.typing.Target):
        """Return an outlier score.

        A high score is indicative of an anomaly. A low score corresponds a normal observation.

        Parameters
        ----------
        x
            A dictionary of features.

        Returns
        -------
        An anomaly score. A high score is indicative of an anomaly. A low score corresponds a
        normal observation.

        """
        return 1.0
