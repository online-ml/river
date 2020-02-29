import abc

from .. import base


class Forecaster(base.Estimator):

    def fit_one(self, y, x=None) -> 'Forecaster':
        """Updates the model.

        Parameters:
            y (float): In the litterature this is called the endogenous variable.
            x (dict): Optional additional features to learn from. In the litterature these are
                called the exogenous variables.

        """

    @abc.abstractmethod
    def forecast(self, horizon: int, xs=None) -> list:
        """Makes forecast at each step of the given horizon.

        Parameters:
            horizon (int): The number of steps ahead to forecast.
            xs (list): The set of . If given, then it's length should be equal to horizon.

        """
