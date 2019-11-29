import collections
import functools
import itertools
import numbers
import numpy as np

from .. import base
from .. import optim
from .. import utils


__all__ = [
    'FFMClassifier',
    'FFMRegressor'
]


class FFM:
    """Field-aware Factorization Machines.

    Parameters:
        n_factors (int): Dimensionality of the factorization or number of latent factors.
        loss (optim.Loss): The loss function to optimize for.
        optimizer (optim.Optimizer): The sequential optimizer used for updating the weights and
            latent factors. Note that the intercept is handled separately.
        intercept (float): Initial intercept value.
        intercept_lr (optim.schedulers.Scheduler or float): Learning rate scheduler used for
            updating the intercept. If a `float` is passed, then an instance of
            `optim.schedulers.Constant` will be used. Setting this to 0 implies that the intercept
            will be not be updated.
        weight_initializer (optim.initializers.Initializer): Weights initialization scheme. Defaults
            to ``optim.initializers.Zeros()``.
        latent_initializer (optim.initializers.Initializer): Latent factors initialization scheme.
            Defaults to ``optim.initializers.Normal(mu=.0, sigma=.1, random_state=random_state)``.
        l1 (float): Amount of L1 regularization used to push weights towards 0.
        l2 (float): Amount of L2 regularization used to push weights towards 0.
        clip_gradient (float): Clips the absolute value of each gradient value.
        random_state (int, ``numpy.random.RandomState`` instance or None): If int, ``random_state``
            is the seed used by the random number generator; if ``RandomState`` instance,
            ``random_state`` is the random number generator; if ``None``, the random number
            generator is the ``RandomState`` instance used by `numpy.random`.

    Attributes:
        weights (collections.defaultdict): The current weights assigned to the features.
        latents (collections.defaultdict): The current latent weights assigned to the features.

    """

    def __init__(self, n_factors, loss, optimizer, intercept, intercept_lr, weight_initializer,
                 latent_initializer, l1, l2, clip_gradient, random_state):
        self.n_factors = n_factors
        self.loss = loss
        self.optimizer = optim.SGD(0.01) if optimizer is None else optimizer
        self.intercept = intercept

        self.intercept_lr = (
            optim.schedulers.Constant(intercept_lr)
            if isinstance(intercept_lr, numbers.Number) else
            intercept_lr
        )

        if weight_initializer:
            self.weight_initializer = weight_initializer
        else:
            self.weight_initializer = optim.initializers.Zeros()

        if latent_initializer:
            self.latent_initializer = latent_initializer
        else:
            self.latent_initializer = optim.initializers.Normal(sigma=.1, random_state=random_state)

        self.l1 = l1
        self.l2 = l2
        self.clip_gradient = clip_gradient

        random_latents = functools.partial(
            self.latent_initializer,
            shape=self.n_factors
        )

        field_latents_dict = functools.partial(
            collections.defaultdict, random_latents
        )

        self.weights = collections.defaultdict(self.weight_initializer)
        self.latents = collections.defaultdict(field_latents_dict)

    def _field(self, j):
        return j.split('_')[0]

    def _raw_dot(self, x):

        # For notational convenience
        w0, w, v = self.intercept, self.weights, self.latents
        field = self._field

        # Start with the intercept
        y_pred = w0

        # Add the unary interactions
        y_pred += utils.math.dot(x, w)

        # Add the pairwise interactions
        y_pred += sum(
            x[j1] * x[j2] * np.dot(v[j1][field(j2)], v[j2][field(j1)])
            for j1, j2 in itertools.combinations(x.keys(), 2)
        )

        return y_pred

    def _latent_gradient(self, j, field, f, x):

        # For notational convenience
        _field = self._field

        # Get derivative terms
        derivative_terms = (
            x[j] * xj2 * self.latents[j2][_field(j)][f]
            for j2, xj2 in x.items()
            if j2 != j and _field(j2) == field
        )

        return sum(derivative_terms)

    def fit_one(self, x, y, sample_weight=1.):

        # For notational convenience
        k, l1, l2 = self.n_factors, self.l1, self.l2
        w0, w, v = self.intercept, self.weights, self.latents
        w0_lr = self.intercept_lr.get(self.optimizer.n_iterations)

        # Some optimizers need to do something before a prediction is made
        self.weights = self.optimizer.update_before_pred(w=w)

        for j in x.keys():
            self.latents[j] = self.optimizer.update_before_pred(w=v[j])

        # Obtain the gradient of the loss with respect to the raw output
        g_loss = self.loss.gradient(y_true=y, y_pred=self._raw_dot(x))

        # Clamp the gradient to avoid numerical instability
        g_loss = utils.math.clamp(g_loss, minimum=-self.clip_gradient, maximum=self.clip_gradient)

        # Apply the sample weight
        g_loss *= sample_weight

        # Update the intercept
        sign = lambda x: -1 if x < 0 else (1 if x > 0 else 0)

        self.intercept -= w0_lr * (g_loss + l1 * sign(w0) + 2. * l2 * w0)

        # Update the weights
        gradient = {
            j: g_loss * xj + l1 * sign(w[j]) + 2. * l2 * w[j]
            for j, xj in x.items()
        }

        self.weights = self.optimizer.update_after_pred(w=w, g=gradient)

        # Update the latent weights
        latent_gradient = {
            j: {
                field: {
                    f: self._latent_gradient(j, field, f, x)
                    for f in range(k)
                }
                for field in v[j].keys()
            }
            for j in x.keys()
        }

        for j in x.keys():
            for field in v[j].keys():
                self.latents[j][field] = self.optimizer.update_after_pred(
                    w=v[j][field],
                    g={
                        f: g_loss * latent_gradient[j][field][f] + l1 * sign(v[j][field][f]) \
                           + 2. * l2 * v[j][field][f]
                        for f in range(k)
                    }
                )

        return self


class FFMRegressor(FFM, base.Regressor):
    """Field-aware Factorization Machines Regressor.

    Parameters:
        n_factors (int): Dimensionality of the factorization or number of latent factors.
        loss (optim.Loss): The loss function to optimize for.
        optimizer (optim.Optimizer): The sequential optimizer used for updating the weights and
            latent factors. Note that the intercept is handled separately.
        intercept (float): Initial intercept value.
        intercept_lr (optim.schedulers.Scheduler or float): Learning rate scheduler used for
            updating the intercept. If a `float` is passed, then an instance of
            `optim.schedulers.Constant` will be used. Setting this to 0 implies that the intercept
            will be not be updated.
        weight_initializer (optim.initializers.Initializer): Weights initialization scheme. Defaults
            to ``optim.initializers.Zeros()``.
        latent_initializer (optim.initializers.Initializer): Latent factors initialization scheme.
            Defaults to ``optim.initializers.Normal(mu=.0, sigma=.1, random_state=random_state)``.
        l1 (float): Amount of L1 regularization used to push weights towards 0.
        l2 (float): Amount of L2 regularization used to push weights towards 0.
        clip_gradient (float): Clips the absolute value of each gradient value.
        random_state (int, ``numpy.random.RandomState`` instance or None): If int, ``random_state``
            is the seed used by the random number generator; if ``RandomState`` instance,
            ``random_state`` is the random number generator; if ``None``, the random number
            generator is the ``RandomState`` instance used by `numpy.random`.

    Attributes:
        weights (collections.defaultdict): The current weights assigned to the features.
        latents (collections.defaultdict): The current latent weights assigned to the features.

    Example:

        ::

            >>> from creme import linear_model

            >>> X_y = (
            ...     ({'user_23': 1, 'movie_Superman': 1, 'time': .12}, 8),
            ...     ({'user_23': 1, 'movie_Terminator': 1, 'time': .13}, 9),
            ...     ({'user_23': 1, 'movie_Star Wars': 1, 'time': .14}, 8),
            ...     ({'user_23': 1, 'movie_Notting Hill': 1, 'time': .15}, 2),
            ...     ({'user_23': 1, 'movie_Harry Potter ': 1, 'time': .16}, 5),
            ...     ({'user_17': 1, 'movie_Superman': 1, 'time': .13}, 8),
            ...     ({'user_17': 1, 'movie_Terminator': 1, 'time': .12}, 9),
            ...     ({'user_17': 1, 'movie_Star Wars': 1, 'time': .16}, 8),
            ...     ({'user_17': 1, 'movie_Notting Hill': 1, 'time': .10}, 2)
            ... )

            >>> model = linear_model.FMRegressor(
            ...     degree=2,
            ...     n_factors=10,
            ...     intercept=5,
            ...     random_state=42,
            ... )

            >>> for x, y in X_y:
            ...     _ = model.fit_one(x, y)

            >>> model.predict_one({'Bob': 1, 'Harry Potter': 1})
            5.240864...

    Note:
        - Using a feature scaler such as `preprocessing.StandardScaler` on non-binary features helps the optimizer to converge.
        - Fields will be infered from feature names by taking everything before first underscore: ``feature_name.split('_')[0]``.

    References:
            1. `Field-aware Factorization Machines for CTR Prediction <https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf>`_

    """

    def __init__(self, n_factors=10, loss=None, optimizer=None, intercept=0., intercept_lr=.01,
                 weight_initializer=None, latent_initializer=None, l1=0., l2=0., clip_gradient=1e12,
                 random_state=None):
        super().__init__(
            n_factors=n_factors,
            loss=optim.losses.Squared() if loss is None else loss,
            optimizer=optimizer,
            intercept=intercept,
            intercept_lr=intercept_lr,
            weight_initializer=weight_initializer,
            latent_initializer=latent_initializer,
            l1=l1,
            l2=l2,
            clip_gradient=clip_gradient,
            random_state=random_state
        )

    def predict_one(self, x):
        return self._raw_dot(x)


class FFMClassifier(FFM, base.BinaryClassifier):
    """Field-aware Factorization Machines Classifier.

    Parameters:
        n_factors (int): Dimensionality of the factorization or number of latent factors.
        loss (optim.Loss): The loss function to optimize for.
        optimizer (optim.Optimizer): The sequential optimizer used for updating the weights and
            latent factors. Note that the intercept is handled separately.
        intercept (float): Initial intercept value.
        intercept_lr (optim.schedulers.Scheduler or float): Learning rate scheduler used for
            updating the intercept. If a `float` is passed, then an instance of
            `optim.schedulers.Constant` will be used. Setting this to 0 implies that the intercept
            will be not be updated.
        weight_initializer (optim.initializers.Initializer): Weights initialization scheme. Defaults
            to ``optim.initializers.Zeros()``.
        latent_initializer (optim.initializers.Initializer): Latent factors initialization scheme.
            Defaults to ``optim.initializers.Normal(mu=.0, sigma=.1, random_state=random_state)``.
        l1 (float): Amount of L1 regularization used to push weights towards 0.
        l2 (float): Amount of L2 regularization used to push weights towards 0.
        clip_gradient (float): Clips the absolute value of each gradient value.
        random_state (int, ``numpy.random.RandomState`` instance or None): If int, ``random_state``
            is the seed used by the random number generator; if ``RandomState`` instance,
            ``random_state`` is the random number generator; if ``None``, the random number
            generator is the ``RandomState`` instance used by `numpy.random`.

    Attributes:
        weights (collections.defaultdict): The current weights assigned to the features.
        latents (collections.defaultdict): The current latent weights assigned to the features.

    Example:

        ::

            >>> from creme import linear_model

            >>> X_y = (
            ...     ({'user_23': 1, 'movie_Superman': 1, 'time': .12}, True),
            ...     ({'user_23': 1, 'movie_Terminator': 1, 'time': .13}, True),
            ...     ({'user_23': 1, 'movie_Star Wars': 1, 'time': .14}, True),
            ...     ({'user_23': 1, 'movie_Notting Hill': 1, 'time': .15}, False),
            ...     ({'user_23': 1, 'movie_Harry Potter ': 1, 'time': .16}, True),
            ...     ({'user_17': 1, 'movie_Superman': 1, 'time': .13}, True),
            ...     ({'user_17': 1, 'movie_Terminator': 1, 'time': .12}, True),
            ...     ({'user_17': 1, 'movie_Star Wars': 1, 'time': .16}, True),
            ...     ({'user_17': 1, 'movie_Notting Hill': 1, 'time': .10}, False)
            ... )

            >>> model = linear_model.FMClassifier(
            ...     degree=2,
            ...     n_factors=10,
            ...     intercept=0,
            ...     random_state=42,
            ... )

            >>> for x, y in X_y:
            ...     _ = model.fit_one(x, y)

            >>> model.predict_one({'Bob': 1, 'Harry Potter': 1})
            True

    Note:
        - Using a feature scaler such as `preprocessing.StandardScaler` on non-binary features helps the optimizer to converge.
        - Fields will be infered from feature names by taking everything before first underscore: ``feature_name.split('_')[0]``.

    References:
            1. `Field-aware Factorization Machines for CTR Prediction <https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf>`_

    """

    def __init__(self, n_factors=10, loss=None, optimizer=None, intercept=0., intercept_lr=.01,
                 weight_initializer=None, latent_initializer=None, l1=0., l2=0., clip_gradient=1e12,
                 random_state=None):
        super().__init__(
            n_factors=n_factors,
            loss=optim.losses.Log() if loss is None else loss,
            optimizer=optimizer,
            intercept=intercept,
            intercept_lr=intercept_lr,
            weight_initializer=weight_initializer,
            latent_initializer=latent_initializer,
            l1=l1,
            l2=l2,
            clip_gradient=clip_gradient,
            random_state=random_state
        )

    def predict_proba_one(self, x):
        p = utils.math.sigmoid(self._raw_dot(x))  # Convert logit to probability
        return {False: 1. - p, True: p}
