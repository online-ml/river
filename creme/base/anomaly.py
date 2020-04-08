import abc

from . import estimator


class AnomalyDetector(estimator.Estimator):

    @abc.abstractmethod
    def fit_one(self, x: dict) -> 'AnomalyDetector':
        """Update the model.

        Parameters:
            x: A dictionary of features.

        """

    @abc.abstractmethod
    def score_one(self, x: dict) -> float:
        """Return an outlier score.

        The range of the score depends on each model. Some models will output anomaly scores
        between 0 and 1, others will not. In any case, the lower the score, the more likely it is
        that `x` is an anomaly.

        Parameters:
            x: A dictionary of features.

        Returns:
            The anomaly score.

        """
