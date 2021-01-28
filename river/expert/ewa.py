import math
import typing

from river import base
from river import linear_model as lm
from river import optim
from river import preprocessing as pp

__all__ = ["EWARegressor"]


class EWARegressor(base.EnsembleMixin, base.Regressor):
    """Exponentially Weighted Average regressor.

    Parameters
    ----------
    regressors
        The regressors to hedge.
    loss
        The loss function that has to be minimized. Defaults to `optim.losses.Squared`.
    learning_rate
        The learning rate by which the model weights are multiplied at each iteration.

    Examples
    --------

    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import expert
    >>> from river import linear_model
    >>> from river import metrics
    >>> from river import optim
    >>> from river import preprocessing
    >>> from river import stream

    >>> optimizers = [
    ...     optim.SGD(0.01),
    ...     optim.RMSProp(),
    ...     optim.AdaGrad()
    ... ]

    >>> for optimizer in optimizers:
    ...
    ...     dataset = datasets.TrumpApproval()
    ...     metric = metrics.MAE()
    ...     model = (
    ...         preprocessing.StandardScaler() |
    ...         linear_model.LinearRegression(
    ...             optimizer=optimizer,
    ...             intercept_lr=.1
    ...         )
    ...     )
    ...
    ...     print(optimizer, evaluate.progressive_val_score(dataset, model, metric))
    SGD MAE: 0.555971
    RMSProp MAE: 0.528284
    AdaGrad MAE: 0.481461

    >>> dataset = datasets.TrumpApproval()
    >>> metric = metrics.MAE()
    >>> hedge = (
    ...     preprocessing.StandardScaler() |
    ...     expert.EWARegressor(
    ...         regressors=[
    ...             linear_model.LinearRegression(optimizer=o, intercept_lr=.1)
    ...             for o in optimizers
    ...         ],
    ...         learning_rate=0.005
    ...     )
    ... )

    >>> evaluate.progressive_val_score(dataset, hedge, metric)
    MAE: 0.494832

    References
    ----------
    [^1]: [Online Learning from Experts: Weighed Majority and Hedge](https://www.shivani-agarwal.net/Teaching/E0370/Aug-2011/Lectures/20-scribe1.pdf)
    [^2]: [Wikipedia page on the multiplicative weight update method](https://www.wikiwand.com/en/Multiplicative_weight_update_method)
    [^3]: [Kivinen, J. and Warmuth, M.K., 1997. Exponentiated gradient versus gradient descent for linear predictors. information and computation, 132(1), pp.1-63.](https://users.soe.ucsc.edu/~manfred/pubs/J36.pdf)

    """

    def __init__(
        self,
        regressors: typing.List[base.Regressor],
        loss: optim.losses.RegressionLoss = None,
        learning_rate=0.5,
    ):
        super().__init__(regressors)
        self.loss = optim.losses.Squared() if loss is None else loss
        self.learning_rate = learning_rate
        self.weights = [1.0] * len(regressors)

    @classmethod
    def _unit_test_params(cls):
        return {
            "regressors": [
                pp.StandardScaler() | lm.LinearRegression(intercept_lr=0.1),
                pp.StandardScaler() | lm.PARegressor(),
            ]
        }

    @property
    def regressors(self):
        return self.models

    def learn_predict_one(self, x, y):

        y_pred_mean = 0.0

        # Make a prediction and update the weights accordingly for each model
        total = 0
        for i, regressor in enumerate(self):
            y_pred = regressor.predict_one(x=x)
            y_pred_mean += self.weights[i] * (y_pred - y_pred_mean) / len(self)
            loss = self.loss(y_true=y, y_pred=y_pred)
            self.weights[i] *= math.exp(-self.learning_rate * loss)
            total += self.weights[i]
            regressor.learn_one(x, y)

        # Normalize the weights so that they sum up to 1
        if total:
            for i, _ in enumerate(self.weights):
                self.weights[i] /= total

        return y_pred_mean

    def learn_one(self, x, y):
        self.learn_predict_one(x, y)
        return self

    def predict_one(self, x):
        return sum(
            model.predict_one(x) * weight for model, weight in zip(self, self.weights)
        )
