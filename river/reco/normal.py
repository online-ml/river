import random

from river import stats

from . import base

__all__ = ["RandomNormal"]


class RandomNormal(base.Recommender):
    """Predicts random values sampled from a normal distribution.

    The parameters of the normal distribution are fitted with running statistics. This is
    equivalent to using `surprise.prediction_algorithms.random_pred.NormalPredictor`.

    This model expects a dict input with a `user` and an `item` entries without any type constraint
    on their values (i.e. can be strings or numbers). Other entries are ignored.

    Parameters
    ----------
    seed
        Randomization seed used for reproducibility.

    Attributes
    ----------
    mean
        stats.Mean
    variance
        stats.Var

    Examples
    --------

    >>> from river import reco

    >>> dataset = (
    ...     ({'user': 'Alice', 'item': 'Superman'}, 8),
    ...     ({'user': 'Alice', 'item': 'Terminator'}, 9),
    ...     ({'user': 'Alice', 'item': 'Star Wars'}, 8),
    ...     ({'user': 'Alice', 'item': 'Notting Hill'}, 2),
    ...     ({'user': 'Alice', 'item': 'Harry Potter'}, 5),
    ...     ({'user': 'Bob', 'item': 'Superman'}, 8),
    ...     ({'user': 'Bob', 'item': 'Terminator'}, 9),
    ...     ({'user': 'Bob', 'item': 'Star Wars'}, 8),
    ...     ({'user': 'Bob', 'item': 'Notting Hill'}, 2)
    ... )

    >>> model = reco.RandomNormal(seed=42)

    >>> for x, y in dataset:
    ...     _ = model.learn_one(x, y)

    >>> model.predict_one({'user': 'Bob', 'item': 'Harry Potter'})
    6.883895

    """

    def __init__(self, seed=None):
        super().__init__()
        self.variance = stats.Var()
        self.mean = stats.Mean()
        self.seed = seed
        self._rng = random.Random(seed)

    def _learn_one(self, user, item, y):
        y_pred = self._predict_one(user, item)
        self.mean.update(y)
        self.variance.update(y)
        return y_pred

    def _predict_one(self, user, item):
        μ = self.mean.get() or 0
        σ = (self.variance.get() or 1) ** 0.5
        return self._rng.gauss(μ, σ)
