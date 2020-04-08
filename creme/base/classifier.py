import abc
import typing

from creme import base

from . import estimator


class Classifier(estimator.Estimator):
    """A classifier."""

    @abc.abstractmethod
    def fit_one(self, x: dict, y: base.typing.ClfTarget) -> 'Classifier':
        """Fits to a set of features `x` and a label `y`.

        Parameters:
            x: A dictionary of features.
            y: A label.

        Returns:
            self

        """

    @abc.abstractmethod
    def predict_proba_one(self, x: dict) -> typing.Dict[base.typing.ClfTarget, float]:
        """Predict the probability of each label for a dictionary of features `x`.

        Parameters:
            x: A dictionary of features.

        Returns:
            A dictionary which associates a probability which each label.

        """

    def predict_one(self, x: dict) -> base.typing.ClfTarget:
        """Predict the label of a set of features `x`.

        Parameters:
            x: A dictionary of features.

        Returns:
            The predicted label.

        """
        y_pred = self.predict_proba_one(x)
        if y_pred:
            return max(y_pred, key=y_pred.get)
        return None


class MultiClassifier(Classifier):
    """A multi-class classifier."""

    @abc.abstractmethod
    def fit_one(self, x: dict, y: base.typing.ClfTarget) -> 'MultiClassifier':
        """Update the model with a set of features `x` and a label `y`.

        Parameters:
            x: A dictionary of features.
            y: A label.

        Returns:
            self

        """


class BinaryClassifier(Classifier):
    """A binary classifier."""

    @abc.abstractmethod
    def fit_one(self, x: dict, y: bool) -> 'BinaryClassifier':
        """Update the model with a set of features `x` and a boolean value `y`.

        Parameters:
            x: A dictionary of features.
            y: A boolean value.

        Returns:
            self

        """

    @abc.abstractmethod
    def predict_proba_one(self, x: dict) -> typing.Dict[bool, float]:
        """Predict the probability of both outcomes for a dictionary of features `x`.

        Parameters:
            x: A dictionary of features.

        Returns:
            A dictionary with the probabilities of `True` and `False`.

        """

    def predict_one(self, x: dict) -> bool:
        """Predict the outcome of a set of features `x`.

        Parameters:
            x: A dictionary of features.

        Returns:
            The predicted outcome.

        """
        y_pred = self.predict_proba_one(x)
        return y_pred[True] > y_pred[False]
