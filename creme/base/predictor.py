import abc

from . import estimator


class Predictor(estimator.Estimator):
    """A predictor.

    A predictor is an estimator that has `predict_one` method. Indeed, not all estimators make
    predictions, although each and every one has a `learn_one` method. This is essentially useful
    for housekeeping.

    """

    @property
    def _supervised(self):
        return True

    @abc.abstractmethod
    def predict_one(self, x):
        pass
