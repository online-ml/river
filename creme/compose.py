from . import base

import numpy as np


class StandardScaleRegressor(base.Regressor):
    """Meta-regressor that rescales the target variable.

    A running mean and standard deviation are maintained and used to modify the target variable
    before feeding it to the underlying regressor. The output of the regressor is unscaled before
    being given to the user.

    """

    def __init__(self, regressor=None, eps=None):
        self.regressor = regressor
        self.count = 0
        self.mean = 0
        self.sos = 0
        self.eps = eps or np.finfo(float).eps

    def _rescale(self, y):
        return (y - self.mean) / (self.eps + self.sos / self.count) ** 0.5

    def _unscale(self, y):
        return y * (self.eps + self.sos / self.count) ** 0.5 + self.mean

    def fit_one(self, x, y):

        self.count += 1
        mean = self.mean
        self.mean += (y - mean) / self.count
        self.sos += (y - mean) * (y - self.mean)

        return self._unscale(self.regressor.fit_one(x, self._rescale(y)))

    def predict_one(self, x):
        return self._unscale(self.regressor.predict_one(x))
