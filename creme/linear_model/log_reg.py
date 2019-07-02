import collections

from .. import base
from .. import optim
from .. import utils


class LogisticRegression(base.BinaryClassifier):
    """Logistic regression.

    Parameters:
        optimizer (optim.Optimizer): The sequential optimizer used to find the best weights.
            Defaults to `optim.VanillaSGD`.
        loss (linear_model.BinaryClassificationLoss): The loss function to optimize for. Defaults
            to `optim.LogLoss`.
        intercept_lr (float): Learning rate used for updating the intercept. Setting this to 0
            means that no intercept will be used, which sometimes helps.
        l2 (float): Amount of L2 regularization used to push weights towards 0.

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
            ...     dataset=datasets.load_breast_cancer(),
            ...     shuffle=True,
            ...     random_state=42
            ... )
            >>> model = (
            ...     preprocessing.StandardScaler() |
            ...     linear_model.LogisticRegression()
            ... )
            >>> metric = metrics.F1()

            >>> model_selection.online_score(X_y, model, metric)
            F1: 0.971989

    """

    def __init__(self, optimizer=None, loss=None, l2=0.0001, intercept_lr=0.01):
        self.optimizer = optim.VanillaSGD(0.05) if optimizer is None else optimizer
        self.loss = optim.LogLoss() if loss is None else loss
        self.intercept_lr = intercept_lr
        self.l2 = l2
        self.weights = collections.defaultdict(float)
        self.intercept = 0.

    def raw_dot(self, x):
        return utils.dot(self.weights, x) + self.intercept

    def fit_one(self, x, y):

        # Some optimizers need to do something before a prediction is made
        self.weights = self.optimizer.update_before_pred(w=self.weights)

        # Obtain the gradient of the loss with respect to the raw output
        g_loss = self.loss.gradient(y_true=y, y_pred=self.raw_dot(x))

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
        self.intercept -= g_loss * self.intercept_lr

        return self

    def predict_proba_one(self, x):
        p = utils.sigmoid(self.raw_dot(x))
        return {True: p, False: 1. - p}
