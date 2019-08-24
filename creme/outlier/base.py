import abc


__all__ = ['OutlierDetector']


class OutlierDetector(abc.ABC):

    @abc.abstractmethod
    def fit_one(self, x):
        """Updates the model"""

    @abc.abstractmethod
    def score_one(self, x):
        """Returns an outlier score."""
