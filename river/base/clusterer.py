from __future__ import annotations

import abc

from . import estimator


class Clusterer(estimator.Estimator):
    """A clustering model."""

    @property
    def _supervised(self) -> bool:
        return False

    @abc.abstractmethod
    def learn_one(self, x: dict) -> None:
        """Update the model with a set of features `x`.

        Parameters
        ----------
        x
            A dictionary of features.

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
