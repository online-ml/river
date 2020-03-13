import abc
import typing

from . import estimator


class Regressor(estimator.Estimator):
    """A regressor."""

    @abc.abstractmethod
    def fit_one(self, x: dict, y: typing.Union[float, int]) -> 'Regressor':
        """Fits to a set of features ``x`` and a real-valued target ``y``.

        Parameters:
            x (dict)
            y (float)

        Returns:
            self: object

        """

    @abc.abstractmethod
    def predict_one(self, x: dict) -> float:
        """Predicts the target value of a set of features ``x``.

        Parameters:
            x (dict)

        Returns:
            float: The prediction.

        """
