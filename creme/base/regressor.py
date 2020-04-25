import abc
import numbers
import typing

from creme import base

from .predictor import Predictor


class Regressor(Predictor):
    """A regressor."""

    @abc.abstractmethod
    def fit_one(self, x: dict, y: base.typing.RegTarget) -> 'Regressor':
        """Fits to a set of features ``x`` and a real-valued target ``y``.

        Parameters:
            x: A dictionary of features.
            y: A numeric target.

        Returns:
            self

        """

    @abc.abstractmethod
    def predict_one(self, x: dict) -> base.typing.RegTarget:
        """Predicts the target value of a set of features ``x``.

        Parameters:
            x: A dictionary of features.

        Returns:
            The prediction.

        """
