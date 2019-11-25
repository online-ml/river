import collections
import functools
import itertools
import numbers

from sklearn import utils as sk_utils

from .. import base
from .. import optim
from .. import stats
from .. import utils


__all__ = [
    'FMClassifier',
    'FMRegressor'
]


class FM:
    """Factorization Machines.

    Parameters:
        degree (int): Polynomial degree or model order.
        n_components (int): Dimensionality of the factorization or number of latent factors.
        init_stdev (float): Standard deviation used to initialize latent factors.
        intercept (float): Initial intercept value.
        loss (optim.Loss): The loss function to optimize for.
        optimizer (optim.Optimizer): The sequential optimizer used for updating the weights. Note
            that the intercept is handled separately.
        l2 (float): Amount of L2 regularization used to push weights towards 0.
        l1 (float): Amount of L1 regularization used to push weights towards 0.
        intercept_lr (optim.schedulers.Scheduler or float): Learning rate scheduler used for
            updating the intercept. If a `float` is passed, then an instance of
            `optim.schedulers.Constant` will be used. Setting this to 0 implies that the intercept
            will be not be updated. Setting this to 0 means that no intercept will be used.
        clip_gradient (float): Clips the absolute value of each gradient value.
        random_state (int, ``numpy.random.RandomState`` instance or None): If int, ``random_state``
            is the seed used by the random number generator; if ``RandomState`` instance,
            ``random_state`` is the random number generator; if ``None``, the random number
            generator is the ``RandomState`` instance used by `numpy.random`.

    Attributes:
        weights (collections.defaultdict): The current weights assigned to the features.
        latents (collections.defaultdict): The current latent weights assigned to the features.
    """

    def __init__(self, degree, n_components, init_stdev, intercept, loss, optimizer, l2, l1,
                 intercept_lr, clip_gradient, random_state):
        self.degree = degree
        self.n_components = n_components
        self.init_stdev = init_stdev
        self.intercept = intercept
        self.loss = optim.losses.Squared() if loss is None else loss
        self.optimizer = optim.SGD(0.01) if optimizer is None else optimizer
        self.l2 = l2
        self.l1 = l1
        self.intercept_lr = (
            optim.schedulers.Constant(intercept_lr)
            if isinstance(intercept_lr, numbers.Number) else
            intercept_lr
        )
        self.clip_gradient = clip_gradient
        self.random_state = sk_utils.check_random_state(random_state)
        self.weights = collections.defaultdict(float)
        self.latents = collections.defaultdict(self._make_random_latent_weights)

    def _make_random_latent_weights(self):
        return {
            f: self.random_state.normal(scale=self.init_stdev)
            for f in range(self.n_components)
        }

    def _calculate_interaction(self, x, combination):
        interaction = functools.reduce(lambda x, y: x * y, (x[j] for j in combination))
        return interaction * sum(
            functools.reduce(lambda x, y: x * y, (self.latents[j][f] for j in combination))
            for f in range(self.n_components)
        )

    def _raw_dot(self, x):

        # For notational convenience
        d, k = self.degree, self.n_components
        w0, w, v = self.intercept, self.weights, self.latents

        # Start with the intercept
        y_pred = w0

        # Add the unary interactions
        y_pred += utils.math.dot(x, w)

        # Add greater than unary interactions
        interactions = (
            self._calculate_interaction(x, combination)
            for l in range(2, d + 1)
            for combination in itertools.combinations(x.keys(), l)
        )

        y_pred += sum(interactions)

        return y_pred

    def fit_one(self, x, y, sample_weight=1.):

        # Obtain the gradient of the loss with respect to the raw output
        g_loss = self.loss.gradient(y_true=y, y_pred=self._raw_dot(x))

        # For notational convenience
        d, k = self.degree, self.n_components
        l1, l2 = self.l1, self.l2
        w0, w, v = self.intercept, self.weights, self.latents
        w0_lr = self.intercept_lr.get(self.optimizer.n_iterations)

        # Update the intercept
        sign = lambda x: -1 if x < 0 else (1 if x > 0 else 0)

        self.intercept -= utils.math.clamp(
            x=w0_lr * (g_loss + l1 * sign(w0) + 2. * l2 * w0),
            minimum=-self.clip_gradient,
            maximum=self.clip_gradient
        ) * sample_weight

        # Update the weights
        gradient = {
            j: utils.math.clamp(
                x=g_loss * xj + 2. * l2 * w[j] + l1 * sign(w[j]),
                minimum=-self.clip_gradient,
                maximum=self.clip_gradient
            ) * sample_weight
            for j, xj in x.items()
        }

        self.weights = self.optimizer.update_after_pred(w=w, g=gradient)

        # Update latent weights
        latent_gradient = collections.defaultdict(lambda: collections.defaultdict(float))

        for l in range(2, d + 1):

            for comb in itertools.combinations(x.keys(), l):

                for j in comb:

                    for f in range(k):

                        product = 1
                        for j2 in comb:
                            product *= x[j2]
                            product *= v[j2][f] if j != j2 else 1

                        latent_gradient[j][f] += product

        for j in x.keys():
            self.latents[j] = self.optimizer.update_after_pred(
                w=v[j],
                g={
                    f: utils.math.clamp(
                        x=g_loss * latent_gradient[j][f] + 2. * l2 * v[j][f] + l1 * sign(v[j][f]),
                        minimum=-self.clip_gradient,
                        maximum=self.clip_gradient
                    ) * sample_weight
                    for f in range(k)
                }
            )

        return self


class FMRegressor(FM, base.Regressor):
    """Factorization Machines Regressor.

    Parameters:
        degree (int): Polynomial degree or model order.
        n_components (int): Dimensionality of the factorization or number of latent factors.
        init_stdev (float): Standard deviation used to initialize latent factors.
        intercept (float): Initial intercept value.
        loss (optim.Loss): The loss function to optimize for.
        optimizer (optim.Optimizer): The sequential optimizer used for updating the weights. Note
            that the intercept is handled separately.
        l2 (float): Amount of L2 regularization used to push weights towards 0.
        l1 (float): Amount of L1 regularization used to push weights towards 0.
        intercept_lr (optim.schedulers.Scheduler or float): Learning rate scheduler used for
            updating the intercept. If a `float` is passed, then an instance of
            `optim.schedulers.Constant` will be used. Setting this to 0 implies that the intercept
            will be not be updated. Setting this to 0 means that no intercept will be used.
        clip_gradient (float): Clips the absolute value of each gradient value.
        random_state (int, ``numpy.random.RandomState`` instance or None): If int, ``random_state``
            is the seed used by the random number generator; if ``RandomState`` instance,
            ``random_state`` is the random number generator; if ``None``, the random number
            generator is the ``RandomState`` instance used by `numpy.random`.

    Attributes:
        weights (collections.defaultdict): The current weights assigned to the features.
        latents (collections.defaultdict): The current latent weights assigned to the features.

    """

    def __init__(self, degree=2, n_components=10, init_stdev=.1, intercept=None, loss=None,
                 optimizer=None, l2=0., l1=0., intercept_lr=.01, clip_gradient=1e12,
                 random_state=None):
        super().__init__(
            degree=degree,
            n_components=n_components,
            init_stdev=init_stdev,
            intercept=intercept,
            loss=optim.losses.Squared() if loss is None else loss,
            optimizer=optim.SGD(0.01) if optimizer is None else optimizer,
            l2=l2,
            l1=l1,
            intercept_lr=(
                optim.schedulers.Constant(intercept_lr)
                if isinstance(intercept_lr, numbers.Number) else
                intercept_lr
            ),
            clip_gradient=clip_gradient,
            random_state=sk_utils.check_random_state(random_state)
        )

    def predict_one(self, x):
        return self._raw_dot(x)


class FMClassifier(FM, base.BinaryClassifier):
    """Factorization Machines Classifier.

    Parameters:
        degree (int): Polynomial degree or model order.
        n_components (int): Dimensionality of the factorization or number of latent factors.
        init_stdev (float): Standard deviation used to initialize latent factors.
        intercept (float): Initial intercept value.
        loss (optim.Loss): The loss function to optimize for.
        optimizer (optim.Optimizer): The sequential optimizer used for updating the weights. Note
            that the intercept is handled separately.
        l2 (float): Amount of L2 regularization used to push weights towards 0.
        l1 (float): Amount of L1 regularization used to push weights towards 0.
        intercept_lr (optim.schedulers.Scheduler or float): Learning rate scheduler used for
            updating the intercept. If a `float` is passed, then an instance of
            `optim.schedulers.Constant` will be used. Setting this to 0 implies that the intercept
            will be not be updated. Setting this to 0 means that no intercept will be used.
        clip_gradient (float): Clips the absolute value of each gradient value.
        random_state (int, ``numpy.random.RandomState`` instance or None): If int, ``random_state``
            is the seed used by the random number generator; if ``RandomState`` instance,
            ``random_state`` is the random number generator; if ``None``, the random number
            generator is the ``RandomState`` instance used by `numpy.random`.

    Attributes:
        weights (collections.defaultdict): The current weights assigned to the features.
        latents (collections.defaultdict): The current latent weights assigned to the features.

    """

    def __init__(self, degree=2, n_components=10, init_stdev=.1, intercept=None, loss=None,
                 optimizer=None, l2=0., l1=0., intercept_lr=.01, clip_gradient=1e12,
                 random_state=None):
        super().__init__(
            degree=degree,
            n_components=n_components,
            init_stdev=init_stdev,
            intercept=intercept,
            loss=optim.losses.Log() if loss is None else loss,
            optimizer=optim.SGD(0.01) if optimizer is None else optimizer,
            l2=l2,
            l1=l1,
            intercept_lr=(
                optim.schedulers.Constant(intercept_lr)
                if isinstance(intercept_lr, numbers.Number) else
                intercept_lr
            ),
            clip_gradient=clip_gradient,
            random_state=sk_utils.check_random_state(random_state)
        )

    def predict_proba_one(self, x):
        p = utils.math.sigmoid(self._raw_dot(x))  # Convert logit to probability
        return {False: 1. - p, True: p}
