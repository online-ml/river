import collections
import copy
import functools
import itertools

from sklearn import utils as sk_utils

from .. import base
from .. import optim
from .. import utils


def powerset(iterable, max_size):
    """powerset([A, B, C], 2, 3) --> (A, B) (A, C) (B, C) (A, B, C)"""
    sizes = range(1, max_size + 1)
    return itertools.chain.from_iterable(
        itertools.combinations(list(iterable), size)
        for size in sizes
    )


class FMRegressor(base.Regressor):
    """

    Example:

    ::

        >>> import datetime as dt
        >>> from creme import compose
        >>> from creme import linear_model
        >>> from creme import preprocessing

        >>> X = [
        ...     {'user': 'Alice', 'movie': 'Titanic', 'date': '2010-1', 'rating': 5},
        ...     {'user': 'Alice', 'movie': 'Notting Hill', 'date': '2010-2', 'rating': 3},
        ...     {'user': 'Alice', 'movie': 'Star Wars', 'date': '2010-4', 'rating': 1},
        ...     {'user': 'Bob', 'movie': 'Star Wars', 'date': '2009-5', 'rating': 4},
        ...     {'user': 'Bob', 'movie': 'Star Trek', 'date': '2009-8', 'rating': 5},
        ...     {'user': 'Charlie', 'movie': 'Titanic', 'date': '2009-9', 'rating': 1},
        ...     {'user': 'Charlie', 'movie': 'Star Wars', 'date': '2009-12', 'rating': 5}
        ... ]

        >>> def get_time_in_months(x):
        ...     t0 = dt.datetime(2008, 11, 1)
        ...     time_in_months = (dt.datetime.strptime(x['date'], '%Y-%m') - t0).days // 31
        ...     return {'time_in_months': time_in_months}

        >>> model = compose.Pipeline([
        ...     ('features', compose.TransformerUnion([
        ...         ('time_in_months', compose.FuncTransformer(get_time_in_months)),
        ...         ('one_hot', compose.Pipeline([
        ...             ('drop_date', preprocessing.Discarder('date')),
        ...             ('one_hot', preprocessing.OneHotEncoder())
        ...         ]))
        ...     ])),
        ...     ('rescale', preprocessing.StandardScaler()),
        ...     ('fm', linear_model.FMRegressor(random_state=42))
        ... ])

        >>> for x in X:
        ...     y = x.pop('rating')
        ...     y_pred = model.fit_predict_one(x, y)
        ...     print(y_pred)
        0.0
        -0.901259...
        0.412813...
        4.826101...
        2.755768...
        2.278344...
        0.406779...

    """

    def __init__(self, degree=2, n_components=5, optimizer=optim.VanillaSGD(0.1),
                 loss=optim.SquaredLoss(), std=1, random_state=None):
        self.degree = degree
        self.n_components = n_components
        self.loss = loss
        self.random_state = sk_utils.check_random_state(random_state)
        self.bias = 0
        self.weights = collections.defaultdict(float)
        rand_normal = functools.partial(self.random_state.normal, scale=std)
        self.latents = collections.defaultdict(lambda: collections.defaultdict(rand_normal))
        self.bias_optimizer = copy.deepcopy(optimizer)
        self.weights_optimizer = copy.deepcopy(optimizer)
        self.latents_optimizer = copy.deepcopy(optimizer)

    def fit_one(self, x, y):

        # Predict the value
        y_pred = self.predict_one(x)

        # Compute the gradient of the loss with respect to the prediction
        loss_gradient = self.loss.gradient(y, y_pred)

        # Update the bias
        self.bias = self.bias_optimizer.update_weights_with_gradient(
            w={'bias': self.bias},
            g={'bias': 1 * loss_gradient}
        )['bias']

        # Update the weights
        self.weights = self.weights_optimizer.update_weights_with_gradient(
            w=self.weights,
            g={i: xi * loss_gradient for i, xi in x.items()}
        )

        # Update the latent weights
        self.latent_sums = {
            j: sum(self.latents[i][j] for i in sorted(x.keys()))
            for j in range(self.n_components)
        }
        for i in x.keys():
            self.latents[i] = self.latents_optimizer.update_weights_with_gradient(
                w=self.latents[i],
                g={
                    j: x[i] * self.latent_sums[j] - self.latents[i][j] * x[i] ** 2
                    for j in range(self.n_components)
                }
            )

        return self

    def predict_one(self, x):

        y_pred = self.bias

        y_pred += utils.dot(self.weights, x)

        for combo in powerset(x.keys(), max_size=self.degree):
            w = utils.chain_dot(*[self.latents[i] for i in combo])
            y_pred += w * utils.prod((x[i] for i in combo))

        return y_pred
