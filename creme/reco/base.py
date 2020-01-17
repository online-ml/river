import abc

from .. import base


class Recommender(base.Estimator):
    """A recommender."""

    def fit_one(self, x, y):
        return self._fit_one(x['user'], x['item'], y)

    def predict_one(self, x):
        return self._predict_one(x['user'], x['item'])

    @abc.abstractmethod
    def _fit_one(self, user, item, y):
        """Fits a ``user``-``item`` pair and a real-valued target ``y``.

        Parameters:
            user (int or str)
            item (int or str)
            y (float)

        Returns:
            self: object

        """

    @abc.abstractmethod
    def _predict_one(self, user, item):
        """Predicts the target value of a set of features ``x``.

        Parameters:
            user (int or str)
            item (int or str)

        Returns:
            float: The prediction.

        """
