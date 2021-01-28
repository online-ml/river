import math

from river import base

__all__ = ["BoxCoxRegressor", "TransformedTargetRegressor"]


class TransformedTargetRegressor(base.Regressor, base.WrapperMixin):
    """Modifies the target before training.

    The user is expected to check that `func` and `inverse_func` are coherent with each other.

    Parameters
    ----------
    regressor
        Regression model to wrap.
    func
        A function modifying the target before training.
    inverse_func
        A function to return to the target's original space.

    Examples
    --------

    >>> import math
    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import linear_model
    >>> from river import meta
    >>> from river import metrics
    >>> from river import preprocessing

    >>> dataset = datasets.TrumpApproval()
    >>> model = (
    ...     preprocessing.StandardScaler() |
    ...     meta.TransformedTargetRegressor(
    ...         regressor=linear_model.LinearRegression(intercept_lr=0.15),
    ...         func=math.log,
    ...         inverse_func=math.exp
    ...     )
    ... )
    >>> metric = metrics.MSE()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    MSE: 8.759624

    """

    def __init__(
        self, regressor: base.Regressor, func: callable, inverse_func: callable
    ):
        self.regressor = regressor
        self.func = func
        self.inverse_func = inverse_func

    @property
    def _wrapped_model(self):
        return self.regressor

    def learn_one(self, x, y):
        self.regressor.learn_one(x, self.func(y))
        return self

    def predict_one(self, x):
        y_pred = self.regressor.predict_one(x)
        return self.inverse_func(y_pred)


class BoxCoxRegressor(TransformedTargetRegressor):
    """Applies the Box-Cox transform to the target before training.

    Box-Cox transform is useful when the target variable is heteroscedastic (i.e. there are
    sub-populations that have different variabilities from others) allowing to transform it towards
    normality.

    The `power` parameter is denoted λ in the literature. If `power` is equal to 0 than the
    Box-Cox transform will be equivalent to a log transform.

    Parameters
    ----------
    regressor
        Regression model to wrap.
    power
        power value to do the transformation.

    Examples
    --------

    >>> import math
    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import linear_model
    >>> from river import meta
    >>> from river import metrics
    >>> from river import preprocessing

    >>> dataset = datasets.TrumpApproval()
    >>> model = (
    ...     preprocessing.StandardScaler() |
    ...     meta.BoxCoxRegressor(
    ...         regressor=linear_model.LinearRegression(intercept_lr=0.2),
    ...         power=0.05
    ...     )
    ... )
    >>> metric = metrics.MSE()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    MSE: 5.898196

    """

    def __init__(self, regressor: base.Regressor, power=1.0):
        super().__init__(
            regressor=regressor,
            func=(lambda y: (y ** power - 1) / power) if power > 0 else math.log,
            inverse_func=(lambda y: (power * y + 1) ** (1 / power))
            if power > 0
            else math.exp,
        )
