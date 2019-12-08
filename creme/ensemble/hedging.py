import math

from .. import base
from .. import optim


__all__ = ['HedgeRegressor']


class HedgeRegressor(base.Ensemble, base.Regressor):
    """Hedge Algorithm for regression.

    The Hedge Algorithm is a special case of the Weighted Majority Algorithm for arbitrary losses.

    Parameters:
        regressors (list of `base.Regressor`): The set of regressor to hedge.
        weights (list of `float`): The initial weight of each model. If ``None`` then a uniform set
            of weights is assumed. This roughly translates to the prior amount of trust we have in
            each model.
        loss (optim.RegressionLoss): The loss function that has to be minimized. Defaults to
            `optim.losses.Squared`.
        learning_rate (float): The learning rate by which the model weights are multiplied at each
            iteration.

    Example:

        ::

            >>> from creme import ensemble
            >>> from creme import linear_model
            >>> from creme import metrics
            >>> from creme import model_selection
            >>> from creme import optim
            >>> from creme import preprocessing
            >>> from creme import stream
            >>> from sklearn import datasets

            >>> optimizers = [
            ...     optim.SGD(0.01),
            ...     optim.RMSProp(),
            ...     optim.AdaGrad()
            ... ]

            >>> for optimizer in optimizers:
            ...
            ...     X_y = stream.iter_sklearn_dataset(
            ...         dataset=datasets.load_boston(),
            ...         shuffle=False
            ...     )
            ...     metric = metrics.MAE()
            ...     model = (
            ...         preprocessing.StandardScaler() |
            ...         linear_model.LinearRegression(
            ...             optimizer=optimizer,
            ...             intercept_lr=.1
            ...         )
            ...     )
            ...
            ...     print(optimizer, model_selection.progressive_val_score(X_y, model, metric))
            SGD MAE: 7.204077
            RMSProp MAE: 3.312495
            AdaGrad MAE: 3.98455

            >>> X_y = stream.iter_sklearn_dataset(
            ...     dataset=datasets.load_boston(),
            ...     shuffle=False
            ... )
            >>> metric = metrics.MAE()
            >>> hedge = (
            ...     preprocessing.StandardScaler() |
            ...     ensemble.HedgeRegressor(
            ...         regressors=[
            ...             linear_model.LinearRegression(optimizer=o, intercept_lr=.1)
            ...             for o in optimizers
            ...         ]
            ...     )
            ... )

            >>> model_selection.progressive_val_score(X_y, hedge, metric)
            MAE: 3.245396

    References:
        1. `Online Learning from Experts: Weighed Majority and Hedge <https://www.shivani-agarwal.net/Teaching/E0370/Aug-2011/Lectures/20-scribe1.pdf>`_
        2. `Multiplicative weight update method <https://www.wikiwand.com/en/Multiplicative_weight_update_method>`_
        3. `Exponentiated Gradient versus GradientDescent for Linear Predictors <https://users.soe.ucsc.edu/~manfred/pubs/J36.pdf>`_

    """

    def __init__(self, regressors, loss=None, learning_rate=.5):
        super().__init__(regressors)
        self.loss = optim.losses.Squared() if loss is None else loss
        self.learning_rate = learning_rate
        self.weights = [1.] * len(regressors)

    @property
    def regressors(self):
        return list(self)

    def fit_predict_one(self, x, y):

        y_pred_mean = 0.

        # Make a prediction and update the weights accordingly for each model
        total = 0
        for i, regressor in enumerate(self):
            y_pred = regressor.predict_one(x=x)
            y_pred_mean += self.weights[i] * (y_pred - y_pred_mean) / len(self)
            loss = self.loss.eval(y_true=y, y_pred=y_pred)
            self.weights[i] *= math.exp(-self.learning_rate * loss)
            total += self.weights[i]
            regressor.fit_one(x, y)

        # Normalize the weights so that they sum up to 1
        if total:
            for i, _ in enumerate(self.weights):
                self.weights[i] /= total

        return y_pred_mean

    def fit_one(self, x, y):
        self.fit_predict_one(x, y)
        return self

    def predict_one(self, x):
        return sum(model.predict_one(x) * weight for model, weight in zip(self, self.weights))
