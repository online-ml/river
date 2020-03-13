import abc

from . import estimator


class Clusterer(estimator.Estimator):
    """A clusterer."""

    @abc.abstractmethod
    def fit_one(self, x: dict) -> 'Clusterer':
        """Fits to a set of features ``x``.

        Parameters:
            x (dict)

        Returns:
            self

        """

    @abc.abstractmethod
    def predict_one(self, x: dict) -> int:
        """Predicts the cluster number of a set of features ``x``.

        Parameters:
            x (dict)

        Returns:
            int

        """
