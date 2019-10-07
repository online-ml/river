from sklearn import utils

from .. import stats

from . import base


__all__ = ['RandomNormal']


class RandomNormal(base.Recommender):
    """Predicts random values sampled from a normal distribution.

    The parameters of the normal distribution are fitted with running statistics. This is
    equivalent to using `surprise.prediction_algorithms.random_pred.NormalPredictor`.

    Parameters:
        random_state (int, ``numpy.random.RandomState`` instance or None): If int, ``random_state``
            is the seed used by the random number generator; if ``RandomState`` instance,
            ``random_state`` is the random number generator; if ``None``, the random number
            generator is the ``RandomState`` instance used by `numpy.random`.

    Attributes:
        variance (stats.Variance)

    """

    def __init__(self, random_state=None):
        super().__init__()
        self.variance = stats.Variance()
        self.random_state = utils.check_random_state(random_state)

    def fit_one(self, r_id, c_id, y):
        y_pred = self.predict_one(r_id, c_id)
        self.variance.update(y)
        return y_pred

    def predict_one(self, r_id, c_id):
        μ = self.variance.mean.get() or 0
        σ = self.variance.get() ** 0.5
        return self.random_state.normal(μ, σ)
