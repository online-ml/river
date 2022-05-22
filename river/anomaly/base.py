import abc

from river import base

__all__ = ["AnomalyDetector"]


class AnomalyDetector(base.Estimator):
    """An anomaly detector."""

    @property
    def _supervised(self):
        return False

    @abc.abstractmethod
    def learn_one(self, x: dict) -> "AnomalyDetector":
        """Update the model.

        Parameters
        ----------
        x
            A dictionary of features.

        Returns
        -------
        self

        """

    @abc.abstractmethod
    def score_one(self, x: dict) -> float:
        """Return an outlier score.

        A high score is indicative of an anomaly. A low score corresponds to a normal observation.

        Parameters
        ----------
        x
            A dictionary of features.

        Returns
        -------
        An anomaly score. A high score is indicative of an anomaly. A low score corresponds a
        normal observation.

        """


class SupervisedAnomalyDetector(base.Estimator):
    """A supervised anomaly detector."""

    @abc.abstractmethod
    def learn_one(self, x: dict, y: base.typing.Target) -> "SupervisedAnomalyDetector":
        """Update the model.

        Parameters
        ----------
        x
            A dictionary of features.

        Returns
        -------
        self

        """

    @abc.abstractmethod
    def score_one(self, x: dict, y: base.typing.Target) -> float:
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


class AnomalyFilter(base.Wrapper, base.Estimator):
    def __init__(self, anomaly_detector: AnomalyDetector):
        self.anomaly_detector = anomaly_detector

    @property
    def _wrapped_model(self):
        return self.anomaly_detector

    @abc.abstractmethod
    def classify(self, score: float) -> bool:
        """Classify an anomaly score as anomalous or not.

        Parameters
        ----------
        score
            An anomaly score to classify.

        Returns
        -------
        A boolean value indicating whether the anomaly score is anomalous or not.

        """

    def score_one(self, *args):
        """Return an outlier score.

        A high score is indicative of an anomaly. A low score corresponds to a normal observation.

        Parameters
        ----------
        args
            Depends on whether the underlying anomaly detector is supervised or not.

        Returns
        -------
        An anomaly score. A high score is indicative of an anomaly. A low score corresponds a
        normal observation.

        """
        return self.anomaly_detector.score_one(*args)

    def learn_one(self, *args):
        """Update the anomaly filter and the underlying anomaly detector.

        Parameters
        ----------
        args
            Depends on whether the underlying anomaly detector is supervised or not.

        Returns
        -------
        self

        """
        score = self.score_one(*args)
        is_anomaly = self.classify(score)
        if not is_anomaly:
            self.anomaly_detector.learn_one(*args)
        return self
