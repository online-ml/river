import copy

from .. import base


__all__ = ['SplitRegressor']


class SplitRegressor(base.Regressor):
    """Runs a different regressor based on the value of a specified attribute.

    Parameters:
        on (str): The feature on which to perform the split.
        models (dict): A mapping between feature values and regressor.
        default_model (base.Regressor): The regressor used for feature values that are not
            specified in ``models``.

    Example:

        ::

            >>> from creme import compose
            >>> from creme import dummy
            >>> from creme import stats

            >>> X = [
            ...     {'key': 'a', 'y': 2},
            ...     {'key': 'a', 'y': 3},
            ...     {'key': 'a', 'y': 4},
            ...     {'key': 'b', 'y': 1},
            ...     {'key': 'b', 'y': 42},
            ...     {'key': 'b', 'y': 1337},
            ...     {'key': 'c', 'y': 6},
            ...     {'key': 'c', 'y': 1},
            ...     {'key': 'c', 'y': 6}
            ... ]

            >>> model = compose.SplitRegressor(
            ...     on='key',
            ...     models={
            ...         'a': dummy.StatisticRegressor(stats.Mean()),
            ...         'b': dummy.StatisticRegressor(stats.Quantile(0.5))
            ...     },
            ...     default_model=dummy.StatisticRegressor(stats.Min())
            ... )

            >>> for x in X:
            ...     y = x.pop('y')
            ...     model = model.fit_one(x, y)

            >>> model.models['a'].statistic.get()
            3.0

            >>> model.predict_one({'key': 'a'})
            3.0

            >>> model.models['b'].statistic.get()
            42

            >>> model.default_model.statistic.get()
            1

    """

    def __init__(self, on, models, default_model):
        self.on = on
        self.models = copy.deepcopy(models)
        self.default_model = copy.deepcopy(default_model)

    def fit_one(self, x, y):
        x = copy.copy(x)
        key = x[self.on]
        x.pop(self.on)

        self.models.get(key, self.default_model).fit_one(x, y)
        return self

    def predict_one(self, x):
        x = copy.copy(x)
        key = x[self.on]
        x.pop(self.on)

        return self.models.get(key, self.default_model).predict_one(x)
