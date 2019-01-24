from sklearn import utils

from .. import stats

from . import base


__all__ = ['RandomNormal']


class RandomNormal(base.Recommender):
    """

    This is equivalent to `surprise.NormalPredictor`.

    """

    def __init__(self, random_state=42):
        super().__init__()
        self.variance = stats.Variance()
        self.rng = utils.check_random_state(random_state)

    def fit_one(self, r_id, c_id, y):
        y_pred = self.predict_one(r_id, c_id)
        self.variance.update(y)
        return y_pred

    def predict_one(self, r_id, c_id):
        μ = self.variance.mean.get()
        σ = self.variance.get() ** 0.5
        return self.rng.normal(μ, σ)
