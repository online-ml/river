import collections

from .. import stats
from .. import optim

from . import base


__all__ = ['SGDBaseline']


class SGDBaseline(base.Recommender):
    """Stochastic gradient descent baseline.

    This is somewhat equivalent to `surprise.prediction_algorithms.baseline_only.BaselineOnly` with
    ``method='sgd'``. The difference is that ``surprise`` always uses plain SGD whereas with
    ``creme`` you can use any SGD method, for example `optim.Adam`.

    Parameters:
        row_optimizer (optim.Optimizer): Optimizer used to tune the row weights.
        col_optimizer (optim.Optimizer): Optimizer used to tune the column weights.
        loss (optim.Loss): Loss function to optimize for.
        l2 (float): regularization amount used to push weights towards 0.
        global_mean (stats.Mean)

    Attributes:
        row_biases (collections.defaultdict)
        col_biases (collections.defaultdict)

    Example:

        ::

            >>> from creme import reco

            >>> ratings = [
            ...     ('Alice', 'Superman', 8),
            ...     ('Alice', 'Terminator', 9),
            ...     ('Alice', 'Star Wars', 8),
            ...     ('Alice', 'Notting Hill', 2),
            ...     ('Alice', 'Harry Potter ', 5),
            ...     ('Bob', 'Superman', 8),
            ...     ('Bob', 'Terminator', 9),
            ...     ('Bob', 'Star Wars', 8),
            ...     ('Bob', 'Notting Hill', 2),
            ... ]

            >>> model = reco.SGDBaseline(
            ...     row_optimizer=optim.SGD(0.005),
            ...     col_optimizer=optim.SGD(0.005)
            ... )

            >>> for user, movie, rating in ratings:
            ...     _ = model.fit_one(user, movie, rating)

            >>> model.predict_one('Bob', 'Harry Potter')
            6.552432...

    """

    def __init__(self, row_optimizer=None, col_optimizer=None, loss=None, l2=0.):
        super().__init__()
        self.row_optimizer = optim.SGD() if row_optimizer is None else row_optimizer
        self.col_optimizer = optim.SGD() if col_optimizer is None else row_optimizer
        self.loss = optim.losses.Squared() if loss is None else row_optimizer
        self.l2 = l2
        self.global_mean = stats.Mean()
        self.row_biases = collections.defaultdict(float)
        self.col_biases = collections.defaultdict(float)

    def fit_one(self, r_id, c_id, y):

        self.row_optimizer.update_before_pred(w=self.row_biases)
        self.col_optimizer.update_before_pred(w=self.col_biases)

        # Predict the value
        y_pred = self.predict_one(r_id, c_id)

        # Compute the gradient of the loss with respect to the prediction
        loss_gradient = self.loss.gradient(y, y_pred)

        # Update the row biases
        r_grad = {r_id: loss_gradient + self.l2 * self.row_biases.get(r_id, 0)}
        self.row_biases = self.row_optimizer.update_after_pred(w=self.row_biases, g=r_grad)

        # Update the row biases
        c_grad = {c_id: loss_gradient + self.l2 * self.col_biases.get(c_id, 0)}
        self.col_biases = self.col_optimizer.update_after_pred(w=self.col_biases, g=c_grad)

        # Update the global mean
        self.global_mean.update(y)

        return y_pred

    def predict_one(self, r_id, c_id):
        return (self.global_mean.get() or 0) + self.row_biases.get(r_id, 0) + self.col_biases.get(c_id, 0)
