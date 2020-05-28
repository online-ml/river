import abc
import typing

from creme import base

from .predictor import Predictor


class Classifier(Predictor):
    """A classifier."""

    @property
    def _is_supervised(self):
        return True

    def predict_one(self, x: dict) -> typing.Optional[base.typing.ClfTarget]:
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

    def predict_proba_one(self, x: dict) -> typing.Dict[base.typing.ClfTarget, float]:
        """Predict the probability of each label for a dictionary of features `x`.

        Parameters:
            x: A dictionary of features.

        Returns:
            A dictionary which associates a probability which each label.

        """
        raise NotImplementedError



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
        raise NotImplementedError
