import collections
import functools

import numpy as np
from sklearn import utils

from .. import stats
from .. import optim

from . import base


__all__ = ['SVD']


class SVD(base.Recommender):
    """ SVD implementation for recommender systems

    Here is an implementation of the Funk SVD, the code is based on the SVD from ``surprise``
    but you can choose the optimizer (in ``surprise``, only SGD is available)

    Parameters:
        n_factors (default=0): The number of latent factors to compute for each vector
        mu (default=0): The mean of the normal distribution for the random weights initialisation
        sigma (default=0.1): The standard deviation of the normal distribution for the random
            weights initialisation
        row_optimizer (optim.Optimizer): Optimizer used to tune the rows weights.
        col_optimizer (optim.Optimizer): Optimizer used to tune the column weights.
        loss (optim.Loss): Loss function to optimize for.
        l2 (float, default=0): regularization amount used to push weights towards 0.
        global_mean (stats.Mean)
        random_state (int, ``numpy.random.RandomState`` instance or None): If int, ``random_state``
            is the seed used by the random number generator; if ``RandomState`` instance,
            ``random_state`` is the random number generator; if ``None``, the random number
            generator is the ``RandomState`` instance used by `numpy.random`.

    Attributes:
        row_biases (collections.defaultdict): The biases of the rows elements (users)
        col_biases (collections.defaultdict): The biases of the columns elements (items)
        row_vec (collections.defaultdict): The latents vectors of the rows elements
            randomly initialized (users)
        col_vec (collections.defaultdict): The latents vectors of the columns elements
            randomly initialized (items)

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
            ...     ('Bob', 'Notting Hill', 2)
            ... ]

            >>> model = reco.SVD(
            ...     n_factors=10,
            ...     row_optimizer=optim.SGD(0.005),
            ...     col_optimizer=optim.SGD(0.005),
            ...     random_state=42
            ... )

            >>> for user, movie, rating in ratings:
            ...     _ = model.fit_one(user, movie, rating)

            >>> model.predict_one('Bob', 'Harry Potter')
            6.552539...

    """

    def __init__(self, n_factors=20, mu=0, sigma=0.1, row_optimizer=None, col_optimizer=None,
                 loss=None, l2=0., random_state=None):

        self.n_factors = n_factors
        self.row_optimizer = optim.SGD() if row_optimizer is None else row_optimizer
        self.col_optimizer = optim.SGD() if col_optimizer is None else row_optimizer
        self.loss = optim.losses.Squared() if loss is None else row_optimizer
        self.l2 = l2
        self.global_mean = stats.Mean()
        self.random_state = utils.check_random_state(random_state)
        self.row_biases = collections.defaultdict(float)
        self.col_biases = collections.defaultdict(float)
        random_weights = functools.partial(
            self.random_state.normal,
            loc=mu,
            scale=sigma,
            size=n_factors
        )
        self.row_vec = collections.defaultdict(random_weights)
        self.col_vec = collections.defaultdict(random_weights)

    def fit_one(self, r_id, c_id, y):

        self.row_optimizer.update_before_pred(w=self.row_biases)
        self.col_optimizer.update_before_pred(w=self.col_biases)

        # Predict the value
        y_pred = self.predict_one(r_id, c_id)

        # Compute the gradient of the loss with respect to the prediction
        loss_gradient = self.loss.gradient(y, y_pred)

        # Update row bias
        r_grad_bias = {r_id: loss_gradient + self.l2 * self.col_biases[r_id]}
        self.row_biases = self.row_optimizer.update_after_pred(self.row_biases, r_grad_bias)

        # Update the row vector
        r_grad_vec = {r_id: loss_gradient * self.col_vec[c_id] + self.l2 * self.row_vec[r_id]}
        self.row_vec = self.row_optimizer.update_after_pred(self.row_vec, r_grad_vec)

        # Update column bias
        c_grad_bias = {c_id: loss_gradient + self.l2 * self.col_biases[c_id]}
        self.col_biases = self.col_optimizer.update_after_pred(self.col_biases, c_grad_bias)

        # Update the column vector
        c_grad_vec = {c_id: loss_gradient * self.row_vec[r_id] + self.l2 * self.col_vec[c_id]}
        self.col_vec = self.col_optimizer.update_after_pred(self.col_vec, c_grad_vec)

        # Update the global mean
        self.global_mean.update(y)

        return y_pred

    def predict_one(self, r_id, c_id):

        # Initialize the prediction to the mean
        y_pred = (self.global_mean.get() or 0)

        # Add the user bias if the user is known
        y_pred += self.row_biases.get(r_id, 0)

        # Add the item bias if the item is known
        y_pred += self.col_biases.get(c_id, 0)

        # Add the dot product of the row and the column if both are known
        y_pred += np.dot(self.row_vec.get(r_id, np.zeros(self.n_factors)),
                         self.col_vec.get(c_id, np.zeros(self.n_factors)))

        return y_pred
