from __future__ import annotations

import abc
import typing
from typing import Any

from river import base

from . import estimator

if typing.TYPE_CHECKING:
    import pandas as pd


class Regressor(estimator.Estimator):
    """A regressor."""

    @abc.abstractmethod
    def learn_one(self, x: dict[base.typing.FeatureName, Any], y: base.typing.RegTarget) -> None:
        """Fits to a set of features `x` and a real-valued target `y`.

        Parameters
        ----------
        x
            A dictionary of features.
        y
            A numeric target.

        """

    @abc.abstractmethod
    def predict_one(self, x: dict[base.typing.FeatureName, Any]) -> base.typing.RegTarget:
        """Predict the output of features `x`.

        Parameters
        ----------
        x
            A dictionary of features.

        Returns
        -------
        The prediction.

        """


class MiniBatchRegressor(Regressor):
    """A regressor that can operate on mini-batches."""

    @abc.abstractmethod
    def learn_many(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Update the model with a mini-batch of features `X` and real-valued targets `y`.

        Parameters
        ----------
        X
            A dataframe of features.
        y
            A series of numbers.

        """

    @abc.abstractmethod
    def predict_many(self, X: pd.DataFrame) -> pd.Series:
        """Predict the outcome for each given sample.

        Parameters
        ----------
        X
            A dataframe of features.

        Returns
        -------
        The predicted outcomes.

        """
