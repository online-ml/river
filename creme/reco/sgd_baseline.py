import collections

from .. import stats
from .. import optim

from . import base


__all__ = ['SGDBaseline']


class SGDBaseline(base.Recommender):
    """

    This is somewhat equivalent to `surprise.BaselineOnly` with `method='sgd'`. The difference is
    that `surprise` always uses plain SGD whereas with `creme` you can use any SGD method, for
    example Adam.

    """

    def __init__(self, row_optimizer=optim.VanillaSGD(), col_optimizer=optim.VanillaSGD(),
                 loss=optim.SquaredLoss(), l2=0):
        super().__init__()
        self.row_optimizer = row_optimizer
        self.col_optimizer = col_optimizer
        self.loss = loss
        self.l2 = l2
        self.global_mean = stats.Mean()
        self.row_biases = collections.defaultdict(lambda: 0.)
        self.col_biases = collections.defaultdict(lambda: 0.)

    def fit_one(self, r_id, c_id, y):

        # Predict the value
        y_pred = self.predict_one(r_id, c_id)

        # Compute the gradient of the loss
        loss_gradient = self.loss.gradient(y, y_pred)

        # Update the row biases
        r_grad = {r_id: loss_gradient + self.l2 * self.row_biases.get(r_id, 0)}
        self.row_biases = self.row_optimizer.update_weights_with_gradient(self.row_biases, r_grad)

        # Update the row biases
        c_grad = {c_id: loss_gradient + self.l2 * self.col_biases.get(c_id, 0)}
        self.col_biases = self.col_optimizer.update_weights_with_gradient(self.col_biases, c_grad)

        # Update the global mean
        self.global_mean.update(y)

        return y_pred

    def predict_one(self, r_id, c_id):
        return self.global_mean.get() + self.row_biases.get(r_id, 0) + self.col_biases.get(c_id, 0)
