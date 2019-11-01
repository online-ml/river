import abc

from .. import base


class Forecaster(base.Estimator):

    @abc.abstractmethod
    def forecast(self, horizon: int, xs=None) -> list:
        """Makes forecast at each step of the given horizon.

        Parameters:
            horizon (int): The number of steps ahead to forecast.
            xs (list): The set of . If given, then it's length should be equal to horizon.

        """
