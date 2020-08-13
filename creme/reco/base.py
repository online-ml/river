import abc
import typing

from creme import base


class Recommender(base.Regressor):
    """A recommender."""

    def fit_one(self, x, y):
        return self._fit_one(x['user'], x['item'], y)

    def predict_one(self, x):
        return self._predict_one(x['user'], x['item'])

    @abc.abstractmethod
    def _fit_one(self, user: typing.Union[str, int], item: typing.Union[str, int],
                 y: float) -> 'Recommender':
        """Fits a `user`-`item` pair and a real-valued target `y`.

        Parameters:
            user: A user ID.
            item: An item ID.
            y: A rating.

        """

    @abc.abstractmethod
    def _predict_one(self, user: typing.Union[str, int], item: typing.Union[str, int]) -> float:
        """Predicts the target value of a set of features `x`.

        Parameters:
            user: A user ID.
            item: An item ID.

        Returns:
            The predicted rating.

        """
