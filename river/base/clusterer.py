import abc

from . import estimator


class Clusterer(estimator.Estimator):
    """A clustering model."""

    @property
    def _supervised(self):
        return False

    @abc.abstractmethod
    def learn_one(self, x: dict, sample_weight: int) -> "Clusterer":
        """Update the model with a set of features `x`.

        Parameters
        ----------
        x
            A dictionary of features.

        sample_weight
            Instance weight. If not provided, uniform weights are assumed.
            Applicability varies depending on the algorithm.

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
