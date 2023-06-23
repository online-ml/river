from __future__ import annotations

from river import base, utils

__all__ = ["PredClipper"]


class PredClipper(base.Wrapper, base.Regressor):
    """Clips the target after predicting.

    Parameters
    ----------
    regressor
        Regressor model for which to clip the predictions.
    y_min
        minimum value.
    y_max
        maximum value.

    Examples
    --------
    >>> from river import linear_model
    >>> from river import preprocessing

    >>> dataset = (
    ...     ({'a': 2, 'b': 4}, 80),
    ...     ({'a': 3, 'b': 5}, 100),
    ...     ({'a': 4, 'b': 6}, 120)
    ... )

    >>> model = preprocessing.PredClipper(
    ...     regressor=linear_model.LinearRegression(),
    ...     y_min=0,
    ...     y_max=200
    ... )

    >>> for x, y in dataset:
    ...     _ = model.learn_one(x, y)

    >>> model.predict_one({'a': -100, 'b': -200})
    0

    >>> model.predict_one({'a': 50, 'b': 60})
    200

    """

    def __init__(self, regressor: base.Regressor, y_min: float, y_max: float):
        self.regressor = regressor
        self.y_min = y_min
        self.y_max = y_max

    @property
    def _wrapped_model(self):
        return self.regressor

    def learn_one(self, x, y, **kwargs):
        self.regressor.learn_one(x=x, y=y, **kwargs)
        return self

    def predict_one(self, x, **kwargs):
        y_pred = self.regressor.predict_one(x=x, **kwargs)
        return utils.math.clamp(y_pred, self.y_min, self.y_max)

    @classmethod
    def _unit_test_params(cls):
        import math

        from river import linear_model

        yield {
            "regressor": linear_model.LinearRegression(),
            "y_min": -math.inf,
            "y_max": math.inf,
        }
