"""
Linear Regression
-----------------
"""
import collections

from .. import base
from .. import optim
from .. import stats

from . import util


__all__ = ['LinearRegression']


class LinearRegression(base.Regressor):
    """Linear regression.

    A linear regression is simply a dot product between some features and some weights. The weights
    are found by using an online optimizer. In the current implementation the intercept is computed
    independently by maintaining a running mean and adding it to each prediction.

    Parameters:
        optimizer (optim.Optimizer): The sequential optimizer used to find the best weights.
        loss (optim.Loss): The loss function to optimize for.
        l2 (float): L2 loss used to push weights towards 0.

    Attributes:
        weights (collections.defaultdict)
        intercept (stats.Mean)

    Example:

        >>> import creme.compose
        >>> import creme.linear_model
        >>> import creme.model_selection
        >>> import creme.optim
        >>> import creme.preprocessing
        >>> import creme.stream
        >>> from sklearn import datasets
        >>> from sklearn import metrics

        >>> X_y = creme.stream.iter_sklearn_dataset(
        ...     load_dataset=datasets.load_boston,
        ...     shuffle=True,
        ...     random_state=42
        ... )
        >>> model = creme.compose.Pipeline([
        ...     ('scale', creme.preprocessing.StandardScaler()),
        ...     ('learn', creme.linear_model.LinearRegression())
        ... ])
        >>> metric = metrics.mean_squared_error

        >>> creme.model_selection.online_score(X_y, model, metric)
        29.561837...

        >>> model.steps[-1][1].intercept.get()
        22.532806...

    """

    def __init__(self, optimizer=optim.VanillaSGD(0.01), loss=optim.SquaredLoss(), l2=0):
        self.optimizer = optimizer
        self.loss = loss
        self.l2 = l2
        self.weights = collections.defaultdict(lambda: 0.)
        self.intercept = stats.Mean()

    def _predict_with_weights(self, x, w):
        return util.dot(x, w) + self.intercept.get()

    def _calc_gradient(self, y_true, y_pred, x, w):
        loss_gradient = self.loss.gradient(y_true, y_pred)
        return {i: xi * loss_gradient + self.l2 * w.get(i, 0) for i, xi in x.items()}

    def fit_one(self, x, y):

        # Update the weights with the error gradient
        self.weights, y_pred = self.optimizer.update_weights(
            x=x,
            y=y,
            w=self.weights,
            f_pred=self._predict_with_weights,
            f_grad=self._calc_gradient
        )

        # The intercept is the running mean of the target
        self.intercept.update(y)

        return y_pred

    def predict_one(self, x):
        return self._predict_with_weights(x, self.weights)
