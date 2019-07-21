import collections

from .. import base
from .. import stats


class Detrender(base.Regressor, base.Wrapper):
    """A simple detrender which centers the target.

    Parameters:
        regressor (base.Regressor)

    """

    def __init__(self, regressor):
        self.regressor = regressor
        self.mean = stats.Mean()

    @property
    def model(self):
        return self.regressor

    def fit_one(self, x, y):
        self.regressor.fit_one(x, y - self.mean.get())
        self.mean.update(y)
        return self

    def predict_one(self, x):
        return self.regressor.predict_one(x) + self.mean.get()


class GroupDetrender(base.Regressor, base.Wrapper):
    """Remove the trend of the target inside each group.

    Parameters:
        regressor (base.Regressor)
        by (str)

    """

    def __init__(self, regressor, by):
        self.regressor = regressor
        self.by = by
        self.means = collections.defaultdict(stats.Mean)

    @property
    def model(self):
        return self.regressor

    def fit_one(self, x, y):
        key = x[self.by]
        self.regressor.fit_one(x, y - self.means[key].get())
        self.means[key].update(y)
        return self

    def predict_one(self, x):
        key = x[self.by]
        return self.regressor.predict_one(x) + self.means[key].get()
