import abc


__all__ = ['OutlierDetector']


class OutlierDetector(abc.ABC):

    @abc.abstractmethod
    def fit_one(self, x):
        """Updates the model"""

    @abc.abstractmethod
    def score_one(self, x):
        """Returns an outlier score.

        The range of the score depends on each model. Some models will output anomaly scores
        between 0 and 1, others will not. In any case, the lower the score, the more likely it is
        that ``x`` is an anomaly.

        """
