from __future__ import annotations

import abc
from typing import Any

from river import base

from . import estimator


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
