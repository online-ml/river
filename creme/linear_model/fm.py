import collections
import copy

from sklearn import utils as sk_utils

from .. import base
from .. import optim
from .. import stats
from .. import utils


class FMRegressor(base.Regressor):
    """Factorization machine for regression.

    Example:

        ::

            >>> from creme import linear_model
            >>> from creme import stream

            >>> features = [
            ... #     Users  |     Movies     |    Movie Ratings   | Time | Last Movies Rated
            ... #    A  B  C | TI  NH  SW  ST | TI   NH   SW   ST  |      | TI  NH  SW  ST
            ...     [1, 0, 0,  1,  0,  0,  0,   0.3, 0.3, 0.3, 0,     13,   0,  0,  0,  0 ],
            ...     [1, 0, 0,  0,  1,  0,  0,   0.3, 0.3, 0.3, 0,     14,   1,  0,  0,  0 ],
            ...     [1, 0, 0,  0,  0,  1,  0,   0.3, 0.3, 0.3, 0,     16,   0,  1,  0,  0 ],
            ...     [0, 1, 0,  0,  0,  1,  0,   0,   0,   0.5, 0.5,   5,    0,  0,  0,  0 ],
            ...     [0, 1, 0,  0,  0,  0,  1,   0,   0,   0.5, 0.5,   8,    0,  0,  1,  0 ],
            ...     [0, 0, 1,  1,  0,  0,  0,   0.5, 0,   0.5, 0,     9,    0,  0,  0,  0 ],
            ...     [0, 0, 1,  0,  0,  1,  0,   0.5, 0,   0.5, 0,     12,   1,  0,  0,  0 ]
            ... ]
            >>> target = [5, 3, 1, 4, 5, 1, 5]

            >>> fm = linear_model.FMRegressor(random_state=42)

            >>> for x, y in stream.iter_array(features[:5], target[:5]):
            ...     fm = fm.fit_one(x, y)

            >>> for x, y in stream.iter_array(features[5:], target[5:]):
            ...     print(fm.predict_one(x), y)
            -8.355825... 1
            -11.355220... 5

    """

    def __init__(self, n_components=5, optimizer=None, loss=None, intercept=None,
                 init_stdev=0.1, random_state=None):
        self.n_components = n_components
        self.loss = optim.losses.Squared() if loss is None else loss
        self.init_stdev = init_stdev
        self.random_state = sk_utils.check_random_state(random_state)

        self.weights = collections.defaultdict(float)
        self.latents = collections.defaultdict(self._make_random_latent_weights)
        if intercept is None or intercept is True:
            intercept = stats.Mean()
        self.intercept = intercept

        optimizer = optim.SGD(0.01) if optimizer is None else optimizer
        self.weights_optimizer = copy.deepcopy(optimizer)
        self.latents_optimizer = copy.deepcopy(optimizer)

    def _make_random_latent_weights(self):
        return {
            f: self.random_state.normal(scale=self.init_stdev)
            for f in range(self.n_components)
        }

    def fit_one(self, x, y):

        # Start by making a prediction
        y_pred = self.predict_one(x)

        # Compute the gradient of the loss with respect to the prediction
        loss_gradient = self.loss.gradient(y, y_pred)

        # Update the weights
        self.weights = self.weights_optimizer.update_after_pred(
            w=self.weights,
            g={j: xj * loss_gradient for j, xj in x.items()}
        )

        # Update the latent weights
        v = self.latents

        vs = {
            f: sum(
                v[j][f] * xj
                for j, xj in x.items()
            )
            for f in range(self.n_components)
        }

        for j, xj in x.items():
            self.latents[j] = self.latents_optimizer.update_after_pred(
                w=self.latents[j],
                g={
                    f: xj * (vs[f] - v[j][f] * xj)
                    for f in range(self.n_components)
                }
            )

        return self

    def predict_one(self, x):

        # For notational convenience
        w0 = self.intercept.get() if self.intercept else 0
        w = self.weights
        v = self.latents

        # Start with the intercept
        y_pred = w0

        # Add the unary interactions
        y_pred = utils.dot(x, w)

        # Add the pairwise interactions
        for f in range(self.n_components):
            y_pred += 0.5 * (
                sum(v[j][f] * xj for j, xj in x.items()) ** 2 -
                sum(v[j][f] ** 2 * xj ** 2 for j, xj in x.items())
            )

        return y_pred
