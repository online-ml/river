from __future__ import annotations

import abc
import typing

from .estimator import Estimator
from .typing import FeatureName, RegTarget


class MultiLabelClassifier(Estimator, abc.ABC):
    """Multi-label classifier."""

    @abc.abstractmethod
    def learn_one(self, x: dict[FeatureName, typing.Any], y: dict[FeatureName, bool]) -> None:
        """Update the model with a set of features `x` and the labels `y`.

        Parameters
        ----------
        x
            A dictionary of features.
        y
            A dictionary of labels.

        """

    def predict_proba_one(
        self, x: dict[FeatureName, typing.Any], **kwargs: typing.Any
    ) -> dict[FeatureName, dict[bool, float]]:
        """Predict the probability of each label appearing given dictionary of features `x`.

        Parameters
        ----------
        x
            A dictionary of features.

        Returns
        -------
        A dictionary that associates a probability which each label.

        """

        # In case the multi-label classifier does not support probabilities
        raise NotImplementedError

    def predict_one(
        self, x: dict[FeatureName, typing.Any], **kwargs: typing.Any
    ) -> dict[FeatureName, bool]:
        """Predict the labels of a set of features `x`.

        Parameters
        ----------
        x
            A dictionary of features.

        Returns
        -------
        The predicted labels.

        """

        probas = self.predict_proba_one(x, **kwargs)

        preds = {}
        for label_id, label_probas in probas.items():
            if not label_probas:
                continue
            preds[label_id] = max(label_probas, key=label_probas.get)  # type: ignore

        return preds


class MultiTargetRegressor(Estimator, abc.ABC):
    """Multi-target regressor."""

    @abc.abstractmethod
    def learn_one(
        self,
        x: dict[FeatureName, typing.Any],
        y: dict[FeatureName, RegTarget],
        **kwargs: typing.Any,
    ) -> None:
        """Fits to a set of features `x` and a real-valued target `y`.

        Parameters
        ----------
        x
            A dictionary of features.
        y
            A dictionary of numeric targets.

        """

    @abc.abstractmethod
    def predict_one(self, x: dict[FeatureName, typing.Any]) -> dict[FeatureName, RegTarget]:
        """Predict the outputs of features `x`.

        Parameters
        ----------
        x
            A dictionary of features.

        Returns
        -------
        The predictions.

        """
