import abc

from . import estimator


class Clusterer(estimator.Estimator):
    """A clustering model."""

    @abc.abstractmethod
    def fit_one(self, x: dict) -> 'Clusterer':
        """Update the model with a set of features `x`.

        Parameters:
            x: A dictionary of features.

        Returns:
            self

        """

    @abc.abstractmethod
    def predict_one(self, x: dict) -> int:
        """Predicts the cluster number for a set of features `x`.

        Parameters:
            x: A dictionary of features.

        Returns:
            A cluster number.

        """
