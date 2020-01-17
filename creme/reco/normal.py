from sklearn import utils

from .. import stats

from . import base


__all__ = ['RandomNormal']


class RandomNormal(base.Recommender):
    """Predicts random values sampled from a normal distribution.

    The parameters of the normal distribution are fitted with running statistics. This is
    equivalent to using `surprise.prediction_algorithms.random_pred.NormalPredictor`. The model
    expect dict inputs containing both a `user` and an `item` entries.

    Parameters:
        random_state (int, ``numpy.random.RandomState`` instance or None): If int, ``random_state``
            is the seed used by the random number generator; if ``RandomState`` instance,
            ``random_state`` is the random number generator; if ``None``, the random number
            generator is the ``RandomState`` instance used by `numpy.random`.

    Attributes:
        mean (stats.Mean)
        variance (stats.Var)

    Example:

        ::

            >>> from creme import reco

            >>> X_y = (
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

            >>> model = reco.RandomNormal(random_state=42)

            >>> for x, y in X_y:
            ...     _ = model.fit_one(x, y)

            >>> model.predict_one({'user': 'Bob', 'item': 'Harry Potter'})
            8.092809...

    """

    def __init__(self, random_state=None):
        super().__init__()
        self.variance = stats.Var()
        self.mean = stats.Mean()
        self.random_state = utils.check_random_state(random_state)

    def _fit_one(self, user, item, y):
        y_pred = self._predict_one(user, item)
        self.mean.update(y)
        self.variance.update(y)
        return y_pred

    def _predict_one(self, user, item):
        μ = self.mean.get() or 0
        σ = (self.variance.get() or 1) ** 0.5
        return self.random_state.normal(μ, σ)
