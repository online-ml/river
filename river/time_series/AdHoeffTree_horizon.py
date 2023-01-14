
from collections import deque
import operator

from river import time_series
from river import tree

__all__ = ["Hoeffding_horizon"]


class AdaptHoeffdingHorizon(time_series.base.Forecaster):
    """

    Parameters
    ----------
    time_series : _type_
        _description_
    """
    def __init__(
        self,
        tree:tree.HoeffdingAdaptiveTreeRegressor
    ):
    
        self.tree = tree
        self._first_values = []
        self._initialized = False

    def learn_one(self, y, x=None):
        self.tree.learn_one(x,y)
        return self

    def predict_one(self, x):
        return
        
    def forecast(self, horizon, xs=None):
        op = operator.add
        return [
            op(
                self.level[-1] + ((h + 1) * self.trend[-1] if self.trend else 0),
                (
                    self.season[-self.seasonality + h % self.seasonality]
                    if self.season
                    else (1 if self.multiplicative else 0)
                ),
            )
            for h in range(horizon)
        ]
