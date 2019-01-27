import numpy as np

from .. import base
from .. import stats


__all__ = ['StandardScaleRegressor']


class StandardScaleRegressor(base.Regressor):
    """Meta-regressor that rescales the target variable.

    A running mean and standard deviation are maintained and used to modify the target variable
    before feeding it to the underlying regressor. The output of the regressor is unscaled before
    being given to the user.

    """

    def __init__(self, regressor=None, eps=None):
        self.regressor = regressor
        """An instance of `creme.base.Regressor`."""
        self.variance = stats.Variance()
        """An instance of `creme.stats.Variance`."""
        self.eps = eps or np.finfo(float).eps
        """Used for avoiding divisions by zero."""

    def _rescale(self, y):
        return (y - self.variance.mean.get()) / (self.variance.get() ** 0.5 + self.eps)

    def _unscale(self, y):
        return y * self.variance.get() ** 0.5 + self.variance.mean.get()

    def fit_one(self, x, y):
        self.variance.update(y)
        return self._unscale(self.regressor.fit_one(x, self._rescale(y)))

    def predict_one(self, x):
        return self._unscale(self.regressor.predict_one(x))
