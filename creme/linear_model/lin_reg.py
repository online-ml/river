import collections

from .. import base
from .. import optim
from .. import stats
from .. import utils


__all__ = ['LinearRegression']


class LinearRegression(base.Regressor):
    """Linear regression.

    A linear regression is simply a dot product between some features and some weights. The weights
    are found by using an online optimizer. The intercept is computed independently by maintaining
    a running mean and adding it to each prediction. Although this isn't the textbook way of doing
    online linear regression, it works just as well if not better.

    Parameters:
        optimizer (optim.Optimizer): The sequential optimizer used to find the best weights.
            Defaults to `optim.VanillaSGD`.
        loss (optim.RegressionLoss): The loss function to optimize for. Defaults to
            `optim.SquaredLoss`.
        intercept (stats.Univariate): Statistic used to estimate the intercept. Defaults to
            `stats.Mean`.
        l2 (float): Amount of L2 regularization used to push weights towards 0.

    Attributes:
        weights (collections.defaultdict)
            Coefficients assigned to each feature.

    Example:

        ::

            >>> from creme import compose
            >>> from creme import linear_model
            >>> from creme import metrics
            >>> from creme import model_selection
            >>> from creme import optim
            >>> from creme import preprocessing
            >>> from creme import stream
            >>> from sklearn import datasets

            >>> X_y = stream.iter_sklearn_dataset(
            ...     dataset=datasets.load_boston(),
            ...     shuffle=True,
            ...     random_state=42
            ... )
            >>> model = (
            ...     preprocessing.StandardScaler() |
            ...     linear_model.LinearRegression()
            ... )
            >>> metric = metrics.MAE()

            >>> model_selection.online_score(X_y, model, metric)
            MAE: 3.668134

            >>> model['LinearRegression'].intercept
            Mean: 22.532806

    Note:
        Using a feature scaler such as `preprocessing.StandardScaler` upstream helps the optimizer
        to converge.

    """

    def __init__(self, optimizer=None, loss=None, intercept=None, l2=0.0001):
        self.optimizer = optim.VanillaSGD(0.01) if optimizer is None else optimizer
        self.loss = optim.SquaredLoss() if loss is None else loss
        self.intercept = stats.Mean() if intercept is None else intercept
        self.l2 = l2
        self.weights = collections.defaultdict(float)

    def fit_one(self, x, y):

        # Some optimizers need to do something before a prediction is made
        self.weights = self.optimizer.update_before_pred(w=self.weights)

        # Obtain the gradient of the loss with respect to the raw output
        g_loss = self.loss.gradient(y_true=y, y_pred=self.predict_one(x))

        # Clip the gradient of the loss to avoid numerical instabilities
        g_loss = utils.clamp(g_loss, -1e12, 1e12)

        # Calculate the gradient
        gradient = {
            i: xi * g_loss + 2. * self.l2 * self.weights.get(i, 0)
            for i, xi in x.items()
        }

        # Update the weights
        self.weights = self.optimizer.update_after_pred(g=gradient, w=self.weights)

        # Update the intercept
        self.intercept.update(y)

        return self

    def predict_one(self, x):
        return utils.dot(self.weights, x) + self.intercept.get()
