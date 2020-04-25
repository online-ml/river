import abc

from .estimator import Estimator


class Predictor(Estimator):

    @abc.abstractmethod
    def predict_one(self, x):
        pass
