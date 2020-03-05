import abc
import typing

from . import estimator


class Classifier(estimator.Estimator):
    """A classifier."""


class BinaryClassifier(Classifier):
    """A binary classifier."""

    @abc.abstractmethod
    def fit_one(self, x: dict, y: bool) -> 'BinaryClassifier':
        """Fits to a set of features ``x`` and a boolean target ``y``.

        Parameters:
            x (dict)
            y (bool)

        Returns:
            self

        """

    @abc.abstractmethod
    def predict_proba_one(self, x: dict) -> typing.Dict[bool, float]:
        """Predicts the probability output of a set of features ``x``.

        Parameters:
            x (dict)

        Returns:
            dict of floats

        """

    def predict_one(self, x: dict) -> bool:
        """Predicts the target value of a set of features ``x``.

        Parameters:
            x (dict)

        Returns:
            bool

        """
        y_pred = self.predict_proba_one(x)
        return y_pred[True] > y_pred[False]


class MultiClassifier(Classifier):
    """A multi-class classifier."""

    @abc.abstractmethod
    def fit_one(self, x: dict, y: typing.Hashable) -> 'MultiClassifier':
        """Fits to a set of features ``x`` and a label ``y``.

        Parameters:
            x (dict)
            y (Label)

        Returns:
            self

        """

    @abc.abstractmethod
    def predict_proba_one(self, x: dict) -> typing.Dict[typing.Hashable, float]:
        """Predicts the probability output of a set of features ``x``.

        Parameters:
            x (dict)

        Returns:
            dict of floats

        """

    def predict_one(self, x: dict) -> typing.Union[typing.Hashable, None]:
        """Predicts the target value of a set of features ``x``.

        Parameters:
            x (dict)

        Returns:
            Label

        """
        y_pred = self.predict_proba_one(x)
        if y_pred:
            return max(y_pred, key=y_pred.get)
        return None
