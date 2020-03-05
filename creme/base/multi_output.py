import abc
import typing

from . import estimator


class MultiOutputEstimator(estimator.Estimator):
    """A multi-output estimator."""


class MultiOutputClassifier(MultiOutputEstimator):
    """A multi-output classifier."""

    @abc.abstractmethod
    def fit_one(self, x: dict, y: typing.Dict[typing.Hashable, typing.Hashable]) -> 'MultiOutputClassifier':
        """Fits to a set of features ``x`` and a set of labels ``y``.

        Parameters:
            x (dict)
            y (dict of Label)

        Returns:
            self

        """

    @abc.abstractmethod
    def predict_proba_one(self, x: dict) -> typing.Dict[typing.Hashable, typing.Dict[typing.Hashable, float]]:
        """Given a set of features ``x``, predicts a the probability of each label for each output.

        Parameters:
            x (dict)

        Returns:

        """

    def predict_one(self, x: dict) -> typing.Dict[typing.Hashable, typing.Hashable]:
        """Given a set of features ``x``, predicts a label for each output.

        Parameters:
            x (dict)

        Returns:

        """
        y_pred = self.predict_proba_one(x)
        return {
            c: max(y_pred[c], key=y_pred[c].get)
            for c in y_pred
        }


class MultiOutputRegressor(MultiOutputEstimator):
    """A multi-output regressor."""

    @abc.abstractmethod
    def fit_one(self, x: dict, y: typing.Dict[typing.Hashable, float]) -> 'MultiOutputRegressor':
        """Fits to a set of features ``x`` and a set of outputs ``y``.

        Parameters:
            x (dict)
            y (dict of Label)

        Returns:
            self

        """

    @abc.abstractmethod
    def predict_one(self, x: dict) -> typing.Dict[typing.Hashable, float]:
        """Given a set of features ``x``, predicts a label for each output.

        Parameters:
            x (dict)

        Returns:
            dict

        """
