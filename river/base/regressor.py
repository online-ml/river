import abc

import pandas as pd

from river import base

from . import estimator


class Regressor(estimator.Estimator):
    """A regressor."""

    @abc.abstractmethod
    def learn_one(self, x: dict, y: base.typing.RegTarget, **kwargs) -> "Regressor":
        """Fits to a set of features `x` and a real-valued target `y`.

        Parameters
        ----------
        x
            A dictionary of features.
        y
            A numeric target.
        kwargs
            Some models might allow/require providing extra parameters, such as sample weights.

        Returns
        -------
        self

        """

    @abc.abstractmethod
    def predict_one(self, x: dict) -> base.typing.RegTarget:
        """Predicts the target value of a set of features `x`.

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
    def learn_many(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "MiniBatchRegressor":
        """Update the model with a mini-batch of features `X` and boolean targets `y`.

        Parameters
        ----------
        X
            A dataframe of features.
        y
            A series of numbers.
        kwargs
            Some models might allow/require providing extra parameters, such as sample weights.

        Returns
        -------
        self

        """

    @abc.abstractmethod
    def predict_many(self, X: pd.DataFrame) -> pd.Series:
        """Predict the outcome for each given sample.

        Parameters
        ---------
        X
            A dataframe of features.

        Returns
        -------
        The predicted outcomes.

        """
