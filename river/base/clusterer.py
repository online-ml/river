import abc

from . import estimator


class Clusterer(estimator.Estimator):
    """A clustering model."""

    @property
    def _supervised(self):
        return False

    @abc.abstractmethod
    def learn_one(self, x: dict) -> "Clusterer":
        """Update the model with a set of features `x`.

        Parameters
        ----------
        x
            A dictionary of features.

        Returns
        -------
        self

        """

    @abc.abstractmethod
    def predict_one(self, x: dict) -> int:
        """Predicts the cluster number for a set of features `x`.

        Parameters
        ----------
        x
            A dictionary of features.

        Returns
        -------
        A cluster number.

        """
