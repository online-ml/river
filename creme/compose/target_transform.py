import numpy as np

from .. import base
from .. import stats


__all__ = ['StandardScaleRegressor']


class StandardScaleRegressor(base.Regressor):
    """Meta-regressor that rescales the target variable nefore fitting.

    A running mean and standard deviation are maintained and used to modify the target variable
    before feeding it to the underlying regressor. The output of the regressor is unscaled before
    being given to the user.

    Parameters:
        regressor (creme.base.Regressor)

    Attributes:
        variance (creme.stats.Variance)
        eps (float): Used for avoiding divisions by zero.

    """

    def __init__(self, regressor=None):
        self.regressor = regressor
        self.variance = stats.Variance()
        self.eps = np.finfo(float).eps

    def _rescale(self, y):
        return (y - self.variance.mean.get()) / (self.variance.get() ** 0.5 + self.eps)

    def _unscale(self, y):
        return y * self.variance.get() ** 0.5 + self.variance.mean.get()

    def fit_one(self, x, y):
        self.variance.update(y)
        return self._unscale(self.regressor.fit_one(x, self._rescale(y)))

    def predict_one(self, x):
        return self._unscale(self.regressor.predict_one(x))
