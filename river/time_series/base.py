from __future__ import annotations

import abc

from river import base

__all__ = ["Forecaster"]


class Forecaster(base.Estimator):
    @property
    def _supervised(self):
        return True

    @abc.abstractmethod
    def learn_one(self, y: float, x: dict | None = None) -> Forecaster:
        """Updates the model.

        Parameters
        ----------
        y
            In the literature this is called the endogenous variable.
        x
            Optional additional features to learn from. In the literature these are called the
            exogenous variables.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def forecast(self, horizon: int, xs: list[dict] | None = None) -> list:
        """Makes forecast at each step of the given horizon.

        Parameters
        ----------
        horizon
            The number of steps ahead to forecast.
        xs
            The set of optional additional features. If given, then it's length should be equal to
            the horizon.

        """
