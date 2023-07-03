from __future__ import annotations

import typing

from river import base


class TargetTransformRegressor(base.Wrapper, base.Regressor):
    """Modifies the target before training.

    The user is expected to check that `func` and `inverse_func` are coherent with each other.

    Parameters
    ----------
    regressor
        Regression model to wrap.
    func
        A function modifying the target before training.
    inverse_func
        A function to return to the target's original space.

    Examples
    --------

    >>> import math
    >>> from river import compose
    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import linear_model
    >>> from river import metrics
    >>> from river import preprocessing

    >>> dataset = datasets.TrumpApproval()
    >>> model = (
    ...     preprocessing.StandardScaler() |
    ...     compose.TargetTransformRegressor(
    ...         regressor=linear_model.LinearRegression(intercept_lr=0.15),
    ...         func=math.log,
    ...         inverse_func=math.exp
    ...     )
    ... )
    >>> metric = metrics.MSE()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    MSE: 10.999752

    """

    def __init__(
        self,
        regressor: base.Regressor,
        func: typing.Callable,
        inverse_func: typing.Callable,
    ):
        self.regressor = regressor
        self.func = func
        self.inverse_func = inverse_func

    @property
    def _wrapped_model(self):
        return self.regressor

    def _update(self, y):
        ...

    def learn_one(self, x, y):
        self._update(y)
        self.regressor.learn_one(x, self.func(y))
        return self

    def predict_one(self, x):
        y_pred = self.regressor.predict_one(x)
        return self.inverse_func(y_pred)
