import collections

from .. import base
from .. import optim

from . import util


__all__ = ['LogisticRegression']


class LogisticRegression(base.BinaryClassifier):
    """Logistic regression for binary classification.

    Parameters:
        optimizer (optim.Optimizer): The sequential optimizer used to find the best weights.
        loss (optim.Loss): The loss function to optimize for.
        l2 (float): regularization amount used to push weights towards 0.

    Attributes:
        weights (collections.defaultdict)

    Example:

    ::

        >>> import creme.compose
        >>> import creme.linear_model
        >>> import creme.model_selection
        >>> import creme.optim
        >>> import creme.preprocessing
        >>> import creme.stream
        >>> from sklearn import datasets
        >>> from sklearn import metrics
        >>> X_y = creme.stream.iter_sklearn_dataset(
        ...     load_dataset=datasets.load_breast_cancer,
        ...     shuffle=True,
        ...     random_state=42
        ... )
        >>> model = creme.compose.Pipeline([
        ...     ('scale', creme.preprocessing.StandardScaler()),
        ...     ('learn', creme.linear_model.LogisticRegression())
        ... ])
        >>> metric = metrics.roc_auc_score

        >>> creme.model_selection.online_score(X_y, model, metric)
        0.990625...

    """

    def __init__(self, optimizer=optim.VanillaSGD(), loss=optim.LogLoss(), l2=0):
        self.optimizer = optimizer
        self.loss = loss
        self.l2 = l2
        self.weights = collections.defaultdict(lambda: 0.)

    def _predict_proba_one_with_weights(self, x, w):
        return util.sigmoid(util.dot(x, w))

    def _calc_gradient(self, y_true, y_pred, x, w):
        loss_gradient = self.loss.gradient(y_true, y_pred)
        return {i: xi * loss_gradient + self.l2 * w.get(i, 0) for i, xi in x.items()}

    def fit_one(self, x, y):

        # Update the weights with the error gradient
        self.weights, y_pred = self.optimizer.update_weights(
            x=x,
            y=y,
            w=self.weights,
            f_pred=self._predict_proba_one_with_weights,
            f_grad=self._calc_gradient
        )

        return y_pred

    def predict_proba_one(self, x):
        return self._predict_proba_one_with_weights(x, self.weights)

    def predict_one(self, x):
        return self.predict_proba_one(x) > 0.5
