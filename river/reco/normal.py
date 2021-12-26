from river import base, stats

from .base import Recommender

__all__ = ["RandomNormal"]


class RandomNormal(Recommender, base.Regressor):
    """Predicts random values sampled from a normal distribution.

    The parameters of the normal distribution are fitted with running statistics. They parameters
    are independent of the user, the item, or the context, and are instead fitted globally. This
    recommender therefore acts as a dummy model that any serious model should easily outperform.

    Parameters
    ----------
    seed
        Random number generation seed. Set this for reproducibility.

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
    6.147299621751425

    """

    def __init__(self, seed=None):
        super().__init__(seed=seed)
        self.variance = stats.Var()
        self.mean = stats.Mean()
        self.seed = seed

    def _learn_user_item(self, user, item, context, reward):
        self.mean.update(reward)
        self.variance.update(reward)
        return self

    def _predict_user_item(self, user, item, context):
        μ = self.mean.get() or 0
        σ = (self.variance.get() or 1) ** 0.5
        return self._rng.gauss(μ, σ)
