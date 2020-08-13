import abc
import typing

from creme import base

from . import estimator


class MultiOutputEstimator(estimator.Estimator):
    """A multi-output estimator."""


class MultiOutputClassifier(MultiOutputEstimator):
    """A multi-output classifier."""

    @abc.abstractmethod
    def fit_one(self, x: dict, y: typing.Dict[typing.Union[str, int], base.typing.ClfTarget]) -> 'MultiOutputClassifier':
        """Fits to a set of features `x` and a set of labels `y`.

        Parameters:
            x: A dictionary of features.
            y: A dictionary with the labels of each output.

        Returns:
            self

        """

    @abc.abstractmethod
    def predict_proba_one(self, x: dict) -> typing.Dict[typing.Union[str, int], typing.Dict[base.typing.ClfTarget, float]]:
        """Given a set of features `x`, predicts a the probability of each label for each output.

        Parameters:
            x: A dictionary of features.

        Returns:
            A nested dictionary which contains the output probabilities of each output.

        """

    def predict_one(self, x: dict) -> typing.Dict[typing.Union[str, int], base.typing.ClfTarget]:
        """Given a set of features `x`, predicts a label for each output.

        Parameters:
            x: A dictionary of features.

        Returns:
            A dictionary with the predicted label for each output.

        """
        y_pred = self.predict_proba_one(x)
        return {
            c: max(y_pred[c], key=y_pred[c].get)
            for c in y_pred
        }


class MultiOutputRegressor(MultiOutputEstimator):
    """A multi-output regressor."""

    @abc.abstractmethod
    def fit_one(self, x: dict, y: typing.Dict[typing.Union[str, int], base.typing.RegTarget]) -> 'MultiOutputRegressor':
        """Fits to a set of features `x` and a set of outputs `y`.

        Parameters:
            x: A dictionary of features.
            y: A dictionary with the target of each output.

        Returns:
            self

        """

    @abc.abstractmethod
    def predict_one(self, x: dict) -> typing.Dict[typing.Union[str, int], base.typing.RegTarget]:
        """Given a set of features `x`, predicts a target for each output.

        Parameters:
            x: A dictionary of features.

        Returns:
            A dictionary with the predicted outcomes of each output.

        """
