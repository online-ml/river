import collections

from .. import base
from .. import losses
from .. import optim

from . import util


__all__ = ['LinearRegression']


class LinearRegression(base.Regressor):
    """
    Examples
    --------

        #!python
        >>> import creme.linear_model
        >>> import creme.pipeline
        >>> import creme.preprocessing
        >>> import creme.stream
        >>> from sklearn import datasets
        >>> from sklearn import metrics

        >>> X_y = creme.stream.iter_sklearn_dataset(
        ...     load_dataset=datasets.load_diabetes,
        ...     shuffle=True,
        ...     random_state=42
        ... )
        >>> model = model = creme.pipeline.Pipeline([
        ...     ('scale', creme.preprocessing.StandardScaler()),
        ...     ('bias', creme.preprocessing.BiasAppender()),
        ...     ('learn', creme.linear_model.LinearRegression())
        ... ])
        >>> metric = metrics.mean_squared_error

        >>> #creme.model_selection.online_score(X_y, model, metric)

    """

    def __init__(self, optimizer=optim.VanillaSGD(0.01), loss=losses.SquaredLoss()):
        self.optimizer = optimizer
        self.loss = loss
        self.weights = collections.defaultdict(lambda: 0.)

    def fit_one(self, x, y):

        # Predict the output of the given features
        y_pred = self.predict_one(x)

        # Compute the error gradient
        loss_gradient = self.loss.gradient(y, y_pred)
        gradient = {i: xi * loss_gradient for i, xi in x.items()}

        # Update the weights with the error gradient
        self.weights = self.optimizer.update_weights(self.weights, gradient)

        return y_pred

    def predict_one(self, x):
        return util.dot(x, self.weights)
