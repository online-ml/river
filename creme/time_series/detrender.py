import collections
import functools

from .. import base
from .. import stats


class Detrender(base.Regressor, base.Wrapper):
    """A linear detrender which centers the target in zero.

    At each ``fit_one`` step, the current mean of ``y`` is substracted from ``y`` before being fed
    to the provided regression model. During the ``predict_one`` step, the current mean is added
    to the prediction of the regression model.

    Parameters:
        regressor (base.Regressor)
        window_size (int): Window size used for calculating the rolling mean. If ``None``, then a
            mean over the whole target data will instead be used.

    """

    def __init__(self, regressor, window_size=None):
        self.regressor = regressor
        self.mean = stats.Mean() if window_size is None else stats.RollingMean(window_size)

    @property
    def _model(self):
        return self.regressor

    def fit_one(self, x, y):
        self.regressor.fit_one(x, y - self.mean.get())
        self.mean.update(y)
        return self

    def predict_one(self, x):
        return self.regressor.predict_one(x) + self.mean.get()


class GroupDetrender(base.Regressor, base.Wrapper):
    """Removes the trend of the target inside each group.

    Parameters:
        regressor (base.Regressor)
        by (str)
        window_size (int): Window size used for calculating each rolling mean. If ``None``, then a
            mean over the whole target data will instead be used.

    """

    def __init__(self, regressor, by, window_size=None):
        self.regressor = regressor
        self.by = by
        self.means = collections.defaultdict(
            stats.Mean if window_size is None else
            functools.partial(stats.RollingMean, window_size)
        )

    @property
    def _model(self):
        return self.regressor

    def fit_one(self, x, y):
        key = x[self.by]
        self.regressor.fit_one(x, y - self.means[key].get())
        self.means[key].update(y)
        return self

    def predict_one(self, x):
        key = x[self.by]
        return self.regressor.predict_one(x) + self.means[key].get()
