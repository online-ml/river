import collections

from .. import base
from .. import losses
from .. import optim

from . import util


__all__ = ['LogisticRegression']


class LogisticRegression(base.BinaryClassifier):

    def __init__(self, optimizer=optim.VanillaSGD(), loss=losses.LogLoss(), l2=0):
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
        return self.predict_proba(x) > 0.5
