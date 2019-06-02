import collections

from .. import base
from .. import optim
from .. import stats
from .. import utils


__all__ = ['LinearRegression']


class bcolors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class LinearRegression(base.Regressor):
    """Linear regression.

    A linear regression is simply a dot product between some features and some weights. The weights
    are found by using an online optimizer. The intercept is computed independently by maintaining
    a running mean and adding it to each prediction. Although this isn't the textbook way of doing
    online linear regression, it works just as well if not better.

    Parameters:
        optimizer (optim.Optimizer): The sequential optimizer used to find the best weights.
            Defaults to `optim.VanillaSGD`.
        loss (optim.RegressionLoss): The loss function to minimize. Defaults to
            `optim.SquaredLoss`.
        l2 (float): Amount of L2 regularization used to push weights towards 0.
        intercept (stats.Univariate): The univariate statistic used to compute the intercept
            online. Defaults to `stats.Mean`.

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
            >>> model = compose.Pipeline([
            ...     ('scale', preprocessing.StandardScaler()),
            ...     ('lin_reg', linear_model.LinearRegression(
            ...         loss=optim.CauchyLoss(),
            ...         optimizer=optim.VanillaSGD(optim.InverseScalingLR(0.1))
            ...     ))
            ... ])
            >>> metric = metrics.MAE()

            >>> model_selection.online_score(X_y, model, metric)
            MAE: 3.584663

            >>> model['lin_reg'].intercept.get()
            22.532806...

    Note:
        Using a feature scaler such as `preprocessing.StandardScaler` upstream helps the optimizer
        to converge.

    """

    def __init__(self, optimizer=None, loss=None, l2=0., intercept=None):
        self.optimizer = optim.VanillaSGD(0.01) if optimizer is None else optimizer
        self.loss = optim.SquaredLoss() if loss is None else loss
        self.l2 = l2
        self.intercept = stats.Mean() if intercept is None else intercept
        self.weights = collections.defaultdict(float)

    def fit_one(self, x, y):

        # Some optimizers need to do something before a prediction is made
        self.weights = self.optimizer.update_before_pred(w=self.weights)

        # Make a prediction for the given features
        y_pred = self.predict_one(x)

        # Compute the gradient w.r.t. each feature
        loss_gradient = self.loss.gradient(y_true=y, y_pred=y_pred)
        gradient = {
            i: xi * loss_gradient + self.l2 * self.weights.get(i, 0)
            for i, xi in x.items()
        }

        # Update the weights by using the gradient
        self.weights = self.optimizer.update_after_pred(g=gradient, w=self.weights)

        # Update the intercept
        if self.intercept:
            self.intercept.update(y)

        return self

    def predict_one(self, x):
        y = utils.dot(x, self.weights)
        if self.intercept:
            y += self.intercept.get()
        return y

    def debug_one(self, x):
        """Prints an explanation of how ``x`` is predicted."""

        def format_weight(w):
            if w > 0:
                return f'{bcolors.GREEN} {w}'
            elif w < 0:
                return f'{bcolors.RED} {w}'
            return f'{bcolors.YELLOW} {w}'

        print(' +\n'.join(
            f'{format_weight(self.weights[i])}{bcolors.ENDC} * {x[i]} ({i})'
            for i in x
        ))
