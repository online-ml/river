import math

from .. import base
from .. import optim


__all__ = ['HedgeClassifier']


class HedgeClassifier(base.BinaryClassifier):
    """A binary classifier that uses the Hedge algorithm to mix models.

    Parameters:
        models (list): The set of models to hedge.
        weights (list): The initial weight of each model. If ``None`` then a uniform set of
            weights is assumed. This roughly translates to the prior amount of trust we have in
            each model.
        loss (optim.Loss): The loss function that has to be minimized.
        learning_rate (float): The learning rate by which the model weights are multiplied at each
            iteration.

    Example:

    ::

        >>> from creme import compose
        >>> from creme import ensemble
        >>> from creme import linear_model
        >>> from creme import model_selection
        >>> from creme import optim
        >>> from creme import preprocessing
        >>> from creme import stream
        >>> from sklearn import datasets
        >>> from sklearn import metrics

        >>> X_y = stream.iter_sklearn_dataset(
        ...     load_dataset=datasets.load_breast_cancer,
        ...     shuffle=False
        ... )
        >>> model = compose.Pipeline([
        ...     ('scale', preprocessing.StandardScaler()),
        ...     ('hedge', ensemble.HedgeClassifier(
        ...         models=[
        ...             linear_model.PassiveAggressiveClassifier(),
        ...             linear_model.PassiveAggressiveIClassifier(),
        ...             linear_model.PassiveAggressiveIIClassifier(),
        ...         ],
        ...         learning_rate=0.9
        ...     ))
        ... ])
        >>> metric = metrics.roc_auc_score

        >>> model_selection.online_score(X_y, model, metric)
        0.980193...

        >>> model.steps[-1][1].weights
        [7.411846...e-129, 9.795003...e-07, 0.999999...]

    References:

    - `Online Learning from Experts: Weighed Majority and Hedge <https://www.shivani-agarwal.net/Teaching/E0370/Aug-2011/Lectures/20-scribe1.pdf>`_

    """

    def __init__(self, models=None, weights=None, loss=optim.LogLoss(), learning_rate=0.5):
        self.models = models
        self.weights = [1 / len(models)] * len(models) if weights is None else weights
        self.loss = loss
        self.learning_rate = learning_rate

    def fit_one(self, x, y):

        # Normalize the weights so that they sum up to 1
        total = sum(self.weights)
        for i, _ in enumerate(self.weights):
            self.weights[i] /= total

        # The consensus is weighted prediction of all the models
        consensus = 0

        for i, (model, weight) in enumerate(zip(self.models, self.weights)):
            y_pred = model.fit_one(x, y)
            consensus += weight * y_pred
            loss = self.loss(y, y_pred)
            self.weights[i] *= math.exp(-self.learning_rate * loss)

        return consensus

    def predict_proba_one(self, x):
        return sum(
            model.predict_proba_one(x) * weight
            for model, weight in zip(self.models, self.weights)
        )

    def predict_one(self, x):
        return self.predict_proba_one(x) > 0.5
