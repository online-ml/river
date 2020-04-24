import abc

from creme import base


class Predictor(base.Estimator):

    @abc.abstractmethod
    def predict_one(self, x):
        pass
