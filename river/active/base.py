from __future__ import annotations

import abc
import random

from river import base

__all__ = ["ActiveLearningClassifier"]


class ActiveLearningClassifier(base.Wrapper, base.Classifier):
    """Base class for active learning classifiers.

    Parameters
    ----------
    classifier
        The classifier to wrap.
    seed
        Random number generator seed for reproducibility.

    """

    def __init__(self, classifier: base.Classifier, seed: int | None = None):
        self.classifier = classifier
        self.seed = seed
        self._rng = random.Random(seed)

    @property
    def _wrapped_model(self):
        return self.classifier

    @abc.abstractmethod
    def _ask_for_label(self, x, y_pred) -> bool:
        ...

    def predict_proba_one(self, x, **kwargs):
        """Predict the probability of each label for `x` and indicate whether a label is needed.

        Parameters
        ----------
        x
            A dictionary of features.

        Returns
        -------
        A dictionary that associates a probability which each label.
        A boolean indicating whether a label is needed.

        """
        y_pred = self.classifier.predict_proba_one(x, **kwargs)
        return y_pred, self._ask_for_label(x, y_pred)

    def predict_one(self, x, **kwargs):
        """Predict the label of `x` and indicate whether a label is needed.

        Parameters
        ----------
        x
            A dictionary of features.

        Returns
        -------
        The predicted label.
        A boolean indicating whether a label is needed.

        """
        y_pred, ask_for_label = self.predict_proba_one(x, **kwargs)
        if y_pred:
            y_pred = max(y_pred, key=y_pred.get)  # type: ignore
        return y_pred, ask_for_label

    def learn_one(self, x, y, **kwargs):
        self.classifier.learn_one(x, y)
        return self
