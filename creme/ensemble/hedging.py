import abc
import collections
import math

from .. import base
from .. import optim


__all__ = ['HedgeBinaryClassifier', 'HedgeRegressor']


class BaseHedge(collections.UserList):

    def __init__(self, models, weights, loss, learning_rate):
        super().__init__()
        self.extend(models)
        self.weights = [1 / len(models)] * len(models) if weights is None else weights
        self.loss = loss
        self.learning_rate = learning_rate

    @abc.abstractmethod
    def _get_prediction(self, model, x):
        """Returns a prediction."""

    def fit_one(self, x, y):

        # Make a prediction and update the weights accordingly for each model
        for i, model in enumerate(self):
            y_pred = self._get_prediction(model, x)
            loss = self.loss(y, y_pred)
            self.weights[i] *= math.exp(-self.learning_rate * loss)
            model.fit_one(x, y)

        # Normalize the weights so that they sum up to 1
        total = sum(self.weights)
        if total:
            for i, _ in enumerate(self.weights):
                self.weights[i] /= total

        return self


class HedgeBinaryClassifier(BaseHedge, base.BinaryClassifier):
    """Hedge Algorithm for binary classification.

    The Hedge Algorithm is a special case of the Weighted Majority Algorithm for arbitrary losses.

    Parameters:
        models (list of `base.BinaryClassifier`): The set of binary classifiers to hedge.
        weights (list of `float`): The initial weight of each model. If ``None`` then a uniform set
            of weights is assumed. This roughly translates to the prior amount of trust we have in
            each model.
        loss (optim.BinaryClassificationLoss): The binary loss function that has to be minimized.
        learning_rate (float): The learning rate by which the model weights are multiplied at each
            iteration.

    Example:

        ::

            >>> from creme import compose
            >>> from creme import ensemble
            >>> from creme import linear_model
            >>> from creme import metrics
            >>> from creme import model_selection
            >>> from creme import preprocessing
            >>> from creme import stream
            >>> from sklearn import datasets

            >>> X_y = stream.iter_sklearn_dataset(
            ...     dataset=datasets.load_breast_cancer(),
            ...     shuffle=False
            ... )
            >>> model = compose.Pipeline([
            ...     ('scale', preprocessing.StandardScaler()),
            ...     ('hedge', ensemble.HedgeBinaryClassifier(
            ...         classifiers=[
            ...             linear_model.PAClassifier(mode=0),
            ...             linear_model.PAClassifier(mode=1),
            ...             linear_model.PAClassifier(mode=2)
            ...         ],
            ...         learning_rate=0.9
            ...     ))
            ... ])
            >>> metric = metrics.F1()

    References:
        1. `Online Learning from Experts: Weighed Majority and Hedge <https://www.shivani-agarwal.net/Teaching/E0370/Aug-2011/Lectures/20-scribe1.pdf>`_
        2. `Multiplicative weight update method <https://www.wikiwand.com/en/Multiplicative_weight_update_method>`_

    """

    def __init__(self, classifiers, weights=None, loss=optim.LogLoss(), learning_rate=0.5):
        super().__init__(
            models=classifiers,
            weights=weights,
            loss=loss,
            learning_rate=learning_rate
        )

    def _get_prediction(self, model, x):
        return model.predict_proba_one(x).get(True, .5)

    def predict_proba_one(self, x):
        y_pred = sum(
            model.predict_proba_one(x).get(True, .5) * weight
            for model, weight in zip(self, self.weights)
        )
        return {False: 1. - y_pred, True: y_pred}


class HedgeRegressor(BaseHedge, base.Regressor):
    """Hedge Algorithm for regression.

    The Hedge Algorithm is a special case of the Weighted Majority Algorithm for arbitrary losses.

    Parameters:
        regressors (list of `base.Regressor`): The set of regressor to hedge.
        weights (list of `float`): The initial weight of each model. If ``None`` then a uniform set
            of weights is assumed. This roughly translates to the prior amount of trust we have in
            each model.
        loss (optim.BinaryClassificationLoss): The binary loss function that has to be minimized.
        learning_rate (float): The learning rate by which the model weights are multiplied at each
            iteration.

    Example:

        ::

            >>> from creme import compose
            >>> from creme import ensemble
            >>> from creme import linear_model
            >>> from creme import metrics
            >>> from creme import model_selection
            >>> from creme import optim
            >>> from creme import preprocessing
            >>> from creme import stream
            >>> from sklearn import datasets

            >>> X_y = stream.iter_sklearn_dataset(
            ...     dataset=datasets.load_boston(),
            ...     shuffle=False
            ... )

            >>> lin_reg = linear_model.LinearRegression
            >>> model = (
            ...     preprocessing.StandardScaler() |
            ...     ensemble.HedgeRegressor(
            ...         regressors=[
            ...             lin_reg(optimizer=optim.SGD(0.01)),
            ...             lin_reg(optimizer=optim.RMSProp()),
            ...             lin_reg(optimizer=optim.AdaGrad())
            ...         ]
            ...     )
            ... )

            >>> metric = metrics.MAE()

            >>> model_selection.online_score(X_y, model, metric)
            MAE: 3.253601

    References:
        1. `Online Learning from Experts: Weighed Majority and Hedge <https://www.shivani-agarwal.net/Teaching/E0370/Aug-2011/Lectures/20-scribe1.pdf>`_
        2. `Multiplicative weight update method <https://www.wikiwand.com/en/Multiplicative_weight_update_method>`_

    """

    def __init__(self, regressors, weights=None, loss=optim.SquaredLoss(), learning_rate=0.5):
        super().__init__(
            models=regressors,
            weights=weights,
            loss=loss,
            learning_rate=learning_rate
        )

    def _get_prediction(self, model, x):
        return model.predict_one(x)

    def predict_one(self, x):
        return sum(model.predict_one(x) * weight for model, weight in zip(self, self.weights))
