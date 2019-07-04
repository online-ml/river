import copy

from .. import base


__all__ = ['GroupRegressor']


class GroupRegressor(base.Regressor, base.Wrapper):
    """Runs a different regressor based on the value of a specified attribute.

    Parameters:
        on (str): The feature on which to perform the split.
        models (dict): A mapping between feature values and regressor.
        default_model (base.Regressor): The regressor used for feature values that are not
            specified in ``models``.

    Example:

        ::

            >>> from creme import datasets
            >>> from creme import ensemble
            >>> from creme import linear_model
            >>> from creme import metrics
            >>> from creme import model_selection
            >>> from creme import preprocessing

            >>> X_y = datasets.load_chick_weights()

            >>> model = ensemble.GroupRegressor(
            ...     on='diet',
            ...     models={
            ...         i: (
            ...             preprocessing.StandardScaler() |
            ...             linear_model.LinearRegression()
            ...         )
            ...         for i in range(1, 5)
            ...     }
            ... )

            >>> model_selection.online_score(X_y, model, metrics.MAE())
            MAE: 26.502444

    """

    def __init__(self, on, models, default_model=None):
        self.on = on
        self.models = copy.deepcopy(models)
        self.default_model = default_model

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
