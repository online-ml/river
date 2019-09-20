import abc
import math

from .. import base
from .. import optim


__all__ = ['HedgeRegressor']


class Hedge(base.Ensemble):

    def __init__(self, models, weights, learning_rate):
        super().__init__(models)
        self.weights = [1] * len(models)
        self.learning_rate = learning_rate

    @abc.abstractmethod
    def _get_loss(self, model, x, y):
        """Returns a prediction."""

    def fit_one(self, x, y):

        # Make a prediction and update the weights accordingly for each model
        total = 0
        for i, model in enumerate(self):
            loss = self._get_loss(model=model, x=x, y=y)
            self.weights[i] *= math.exp(-self.learning_rate * loss)
            total += self.weights[i]
            model.fit_one(x, y)

        # Normalize the weights so that they sum up to 1
        if total:
            for i, _ in enumerate(self.weights):
                self.weights[i] /= total

        return self


class HedgeRegressor(Hedge, base.Regressor):
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
            ...     print(optimizer, model_selection.online_score(X_y, model, metric))
            SGD MAE: 7.203527
            RMSProp MAE: 3.312368
            AdaGrad MAE: 3.984558

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

            >>> model_selection.online_score(X_y, hedge, metric)
            MAE: 3.245318

    References:
        1. `Online Learning from Experts: Weighed Majority and Hedge <https://www.shivani-agarwal.net/Teaching/E0370/Aug-2011/Lectures/20-scribe1.pdf>`_
        2. `Multiplicative weight update method <https://www.wikiwand.com/en/Multiplicative_weight_update_method>`_
        3. `Exponentiated Gradient versus GradientDescent for Linear Predictors <https://users.soe.ucsc.edu/~manfred/pubs/J36.pdf>`_

    """

    def __init__(self, regressors, weights=None, loss=None, learning_rate=0.5):
        super().__init__(
            models=regressors,
            weights=weights,
            learning_rate=learning_rate
        )
        self.loss = optim.losses.Squared() if loss is None else loss

    def _get_loss(self, model, x, y):
        return self.loss.eval(y_true=y, y_pred=model.predict_one(x))

    def predict_one(self, x):
        return sum(model.predict_one(x) * weight for model, weight in zip(self, self.weights))
