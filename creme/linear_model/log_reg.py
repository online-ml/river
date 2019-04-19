import collections

from .. import base
from .. import optim
from .. import utils


__all__ = ['LogisticRegression']


class LogisticRegression(base.BinaryClassifier):
    """Logistic regression for binary classification.

    Parameters:
        optimizer (optim.Optimizer): The sequential optimizer used to find the best weights.
        loss (optim.BinaryClassificationLoss): The loss function to optimize for.
        l2 (float): Amount of L2 regularization used to push weights towards 0.

    Attributes:
        weights (collections.defaultdict)

    Example:

        ::

            >>> from creme import compose
            >>> from creme import linear_model
            >>> from creme import metrics
            >>> from creme import model_selection
            >>> from creme import preprocessing
            >>> from creme import stream
            >>> from sklearn import datasets

            >>> X_y = stream.iter_sklearn_dataset(
            ...     load_dataset=datasets.load_breast_cancer,
            ...     shuffle=True,
            ...     random_state=42
            ... )
            >>> model = compose.Pipeline([
            ...     ('scale', preprocessing.StandardScaler()),
            ...     ('learn', linear_model.LogisticRegression())
            ... ])
            >>> metric = metrics.F1Score()

            >>> model_selection.online_score(X_y, model, metric)
            F1Score: 0.964739

    """

    def __init__(self, optimizer=None, loss=None, l2=0):
        self.optimizer = optim.VanillaSGD(0.01) if optimizer is None else optimizer
        self.loss = optim.LogLoss() if loss is None else loss
        self.l2 = l2
        self.weights = collections.defaultdict(float)

    def fit_one(self, x, y):

        # Some optimizers need to do something before a prediction is made
        self.weights = self.optimizer.update_before_pred(w=self.weights)

        # Make a prediction for the given features
        y_pred = self.predict_proba_one(x)[True]

        # Compute the gradient w.r.t. each feature
        loss_gradient = self.loss.gradient(y_true=y, y_pred=y_pred)
        gradient = {
            i: xi * loss_gradient + self.l2 * self.weights.get(i, 0)
            for i, xi in x.items()
        }

        # Update the weights by using the gradient
        self.weights = self.optimizer.update_after_pred(g=gradient, w=self.weights)

        return self

    def predict_proba_one(self, x):
        y_pred = utils.sigmoid(utils.dot(x, self.weights))
        return {False: 1. - y_pred, True: y_pred}
