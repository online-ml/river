import abc
import collections
import math

from .. import base
from .. import optim


__all__ = ['HedgeClassifier', 'HedgeRegressor']


class BaseHedge(collections.UserDict):

    def __init__(self, models, weights, loss, learning_rate):
        super().__init__()
        self.update(dict(enumerate(models)))
        weights = [1 / len(models)] * len(models) if weights is None else weights
        self.weights = dict(zip(self.keys(), weights))
        self.loss = loss
        self.learning_rate = learning_rate

    @abc.abstractmethod
    def _get_prediction(self, model, x):
        """Returns a prediction."""

    def fit_one(self, x, y):

        # Make a prediction and update the weights accordingly for each model
        for i, model in self.items():
            y_pred = self._get_prediction(model, x)
            loss = self.loss(y, y_pred)
            self.weights[i] *= math.exp(-self.learning_rate * loss)
            model.fit_one(x, y)

        # Normalize the weights so that they sum up to 1
        total = sum(self.weights.values())
        for i in self.weights:
            self.weights[i] /= total

        return self


class HedgeClassifier(BaseHedge, base.BinaryClassifier):
    """Hedge Algorithm for classification.

    The Hedge Algorithm is a special case of the Weighted Majority Algorithm for arbitrary losses.

    Parameters:
        models (list): The set of binary classifiers to hedge.
        weights (list): The initial weight of each model. If ``None`` then a uniform set of
            weights is assumed. This roughly translates to the prior amount of trust we have in
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
            ...     ('hedge', ensemble.HedgeClassifier(
            ...         models=[
            ...             linear_model.PAClassifier(mode=0),
            ...             linear_model.PAClassifier(mode=1),
            ...             linear_model.PAClassifier(mode=2),
            ...         ],
            ...         learning_rate=0.9
            ...     ))
            ... ])
            >>> metric = metrics.F1()

            >>> model_selection.online_score(X_y, model, metric)
            F1: 0.944369

            >>> model['hedge'].weights
            {0: 0.999999..., 1: 1.127738...e-07, 2: 3.705275...e-17}

    References:
        1. `Online Learning from Experts: Weighed Majority and Hedge <https://www.shivani-agarwal.net/Teaching/E0370/Aug-2011/Lectures/20-scribe1.pdf>`_
        2. `Multiplicative weight update method <https://www.wikiwand.com/en/Multiplicative_weight_update_method>`_

    """

    def __init__(self, models, weights=None, loss=optim.LogLoss(), learning_rate=0.5):
        super().__init__(
            models=models,
            weights=weights,
            loss=loss,
            learning_rate=learning_rate
        )

    def _get_prediction(self, model, x):
        return model.predict_proba_one(x)[True]

    def predict_proba_one(self, x):
        y_pred = sum(
            model.predict_proba_one(x)[True] * self.weights[i]
            for i, model in self.items()
        )
        return {False: 1 - y_pred, True: y_pred}


class HedgeRegressor(BaseHedge, base.Regressor):
    """Hedge Algorithm for regression.

    The Hedge Algorithm is a special case of the Weighted Majority Algorithm for arbitrary losses.

    Parameters:
        models (list): The set of binary classifiers to hedge.
        weights (list): The initial weight of each model. If ``None`` then a uniform set of
            weights is assumed. This roughly translates to the prior amount of trust we have in
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
            >>> model = compose.Pipeline([
            ...     ('scale', preprocessing.StandardScaler()),
            ...     ('hedge', ensemble.HedgeRegressor(
            ...         models=[
            ...             linear_model.LinearRegression(optim.VanillaSGD()),
            ...             linear_model.LinearRegression(optim.RMSProp()),
            ...             linear_model.LinearRegression(optim.AdaGrad()),
            ...         ],
            ...         learning_rate=0.9
            ...     ))
            ... ])
            >>> metric = metrics.MAE()

            >>> model_selection.online_score(X_y, model, metric)
            MAE: 3.388058

    References:
        1. `Online Learning from Experts: Weighed Majority and Hedge <https://www.shivani-agarwal.net/Teaching/E0370/Aug-2011/Lectures/20-scribe1.pdf>`_
        2. `Multiplicative weight update method <https://www.wikiwand.com/en/Multiplicative_weight_update_method>`_

    """

    def __init__(self, models, weights=None, loss=optim.SquaredLoss(), learning_rate=0.5):
        super().__init__(
            models=models,
            weights=weights,
            loss=loss,
            learning_rate=learning_rate
        )

    def _get_prediction(self, model, x):
        return model.predict_one(x)

    def predict_one(self, x):
        return sum(model.predict_one(x) * self.weights[i] for i, model in self.items())
