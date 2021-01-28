import collections
import functools
import typing

from river import base, stats


class Detrender(base.Regressor, base.WrapperMixin):
    """A linear detrender which centers the target in zero.

    At each `learn_one` step, the current mean of `y` is subtracted from `y` before being fed
    to the provided regression model. During the `predict_one` step, the current mean is added
    to the prediction of the regression model.

    Parameters
    ----------
    regressor
    window_size
        Window size used for calculating the rolling mean. If `None`, then a mean over the whole
        target data will instead be used.

    """

    def __init__(self, regressor: base.Regressor, window_size: int = None):
        self.regressor = regressor
        self.mean = (
            stats.Mean() if window_size is None else stats.RollingMean(window_size)
        )

    @property
    def _wrapped_model(self):
        return self.regressor

    def learn_one(self, x, y):
        self.regressor.learn_one(x, y - self.mean.get())
        self.mean.update(y)
        return self

    def predict_one(self, x):
        return self.regressor.predict_one(x) + self.mean.get()


class GroupDetrender(base.Regressor, base.WrapperMixin):
    """Removes the trend of the target inside each group.

    Parameters
    ----------
    regressor
    by
    window_size
        Window size used for calculating each rolling mean. If `None`, then a mean over the whole
        target data will instead be used.

    """

    def __init__(self, regressor: base.Regressor, by: str, window_size: int = None):
        self.regressor = regressor
        self.by = by
        self.means: typing.DefaultDict[
            typing.Any, stats.Univariate
        ] = collections.defaultdict(
            stats.Mean
            if window_size is None
            else functools.partial(stats.RollingMean, window_size)
        )

    @property
    def _wrapped_model(self):
        return self.regressor

    def learn_one(self, x, y):
        key = x[self.by]
        self.regressor.learn_one(x, y - self.means[key].get())
        self.means[key].update(y)
        return self

    def predict_one(self, x):
        key = x[self.by]
        return self.regressor.predict_one(x) + self.means[key].get()
