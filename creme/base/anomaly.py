import abc

from . import estimator


class AnomalyDetector(estimator.Estimator):

    @abc.abstractmethod
    def learn_one(self, x: dict) -> 'AnomalyDetector':
        """Update the model.

        Parameters:
            x: A dictionary of features.

        Returns:
            self

        """

    @abc.abstractmethod
    def score_one(self, x: dict) -> float:
        """Return an outlier score.

        A high score is indicative of an anomaly. A low score indicates a normal observation.

        Parameters:
            x: A dictionary of features.

        Returns:
            The anomaly score.

        """
