import collections

from .. import base
from .. import optim
from .. import stats
from .. import utils


__all__ = ['LinearRegression']


class NilIntercept:

    def update(self, x):
        return self

    def get(self):
        return 0


class LinearRegression(base.Regressor):
    """Linear regression.

    A linear regression is simply a dot product between some features and some weights. The weights
    are found by using an online optimizer. The intercept is computed independently by maintaining
    a running mean and adding it to each prediction. Although this isn't the textbook way of doing
    online linear regression, it works just as well if not better.

    Parameters:
        optimizer (optim.Optimizer): The sequential optimizer used to find the best weights.
        loss (optim.Loss): The loss function to optimize for.
        l2 (float): L2 loss used to push weights towards 0.
        intercept (stats.RunningStatistic): The statistic used to compute the intercept online.

    Attributes:
        weights (collections.defaultdict)

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
        ...     load_dataset=datasets.load_boston,
        ...     shuffle=True,
        ...     random_state=42
        ... )
        >>> model = compose.Pipeline([
        ...     ('scale', preprocessing.StandardScaler()),
        ...     ('lin_reg', linear_model.LinearRegression())
        ... ])
        >>> metric = metrics.MSE()

        >>> model_selection.online_score(X_y, model, metric)
        MSE: 29.58786

        >>> model['lin_reg'].intercept.get()
        22.532806...

    """

    def __init__(self, optimizer=None, loss=None, l2=0, intercept=None):
        self.optimizer = optim.VanillaSGD(0.01) if optimizer is None else optimizer
        self.loss = optim.SquaredLoss() if loss is None else loss
        self.l2 = l2
        self.weights = collections.defaultdict(float)
        self.intercept = {
            False: NilIntercept(),
            True: stats.Mean(),
            None: stats.Mean(),
        }.get(intercept, intercept)

    def _predict_with_weights(self, x, w):
        return utils.dot(x, w) + self.intercept.get()

    def _calc_gradient(self, y_true, y_pred, loss, x, w):
        loss_gradient = loss.gradient(y_true=y_true, y_pred=y_pred)
        return {i: xi * loss_gradient + self.l2 * w.get(i, 0) for i, xi in x.items()}

    def fit_one(self, x, y):
        self.fit_predict_one(x, y)
        return self

    def predict_one(self, x):
        return self._predict_with_weights(x, self.weights)

    def fit_predict_one(self, x, y):

        # Update the weights with the error gradient
        self.weights, y_pred = self.optimizer.update_weights(
            x=x,
            y=y,
            w=self.weights,
            loss=self.loss,
            f_pred=self._predict_with_weights,
            f_grad=self._calc_gradient
        )

        # The intercept is the running mean of the target
        self.intercept.update(y)

        return y_pred
