import collections
import functools
import itertools
import numbers

from .. import base
from .. import optim
from .. import utils


__all__ = [
    'FMClassifier',
    'FMRegressor'
]


class FM:
    """Factorization Machines.

    Parameters:
        degree (int): Polynomial degree or model order.
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

    def __init__(self, degree, n_factors, loss, optimizer, intercept, intercept_lr,
                 weight_initializer, latent_initializer, l1, l2, clip_gradient, random_state):
        self.degree = degree
        self.n_factors = n_factors
        self.loss = loss
        self.optimizer = optim.SGD(0.01) if optimizer is None else optimizer
        self.intercept = intercept

        self.intercept_lr = (
            optim.schedulers.Constant(intercept_lr)
            if isinstance(intercept_lr, numbers.Number) else
            intercept_lr
        )

        if weight_initializer is None:
            weight_initializer = optim.initializers.Zeros()
        self.weight_initializer = weight_initializer

        if latent_initializer is None:
            latent_initializer = optim.initializers.Normal(sigma=.1, random_state=random_state)
        self.latent_initializer = latent_initializer

        self.l1 = l1
        self.l2 = l2
        self.clip_gradient = clip_gradient

        random_latents = functools.partial(
            self.latent_initializer,
            shape=self.n_factors
        )

        self.weights = collections.defaultdict(self.weight_initializer)
        self.latents = collections.defaultdict(random_latents)

    def _transform(self, x):
        return dict((f'{j}_{xj}', 1) if isinstance(xj, str) else (j, xj) for j, xj in x.items())

    def _calculate_interaction(self, x, combination):
        interaction = functools.reduce(lambda x, y: x * y, (x[j] for j in combination))
        return interaction * sum(
            functools.reduce(lambda x, y: x * y, (self.latents[j][f] for j in combination))
            for f in range(self.n_factors)
        )

    def _raw_dot(self, x):

        # Start with the intercept
        y_pred = self.intercept

        # Add the unary interactions
        y_pred += utils.math.dot(x, self.weights)

        # Add greater than unary interactions
        y_pred += sum(
            self._calculate_interaction(x, combination)
            for l in range(2, self.degree + 1)
            for combination in itertools.combinations(x.keys(), l)
        )

        return y_pred

    def _latent_gradient(self, j, f, x):

            # Get all interaction combinations with j
            combinations = (
                comb
                for l in range(2, self.degree + 1)
                for comb in itertools.combinations(x.keys(), l)
                if j in comb
            )

            # Get derivative terms
            derivative_terms = (
                functools.reduce(
                    lambda x, y: x * y, itertools.chain(
                        (x[j2] for j2 in comb),
                        (self.latents[j2][f] for j2 in comb if j2 != j)
                    )
                )
                for comb in combinations
            )

            return sum(derivative_terms)

    def fit_one(self, x, y, sample_weight=1.):

        # One hot encode string modalities
        x = self._transform(x)

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
                f: self._latent_gradient(j, f, x)
                for f in range(k)
            }
            for j in x.keys()
        }

        for j in x.keys():
            self.latents[j] = self.optimizer.update_after_pred(
                w=v[j],
                g={
                    f: g_loss * latent_gradient[j][f] + l1 * sign(v[j][f]) + 2. * l2 * v[j][f]
                    for f in range(k)
                }
            )

        return self


class FMRegressor(FM, base.Regressor):
    """Factorization Machines Regressor.

    Parameters:
        degree (int): Polynomial degree or model order.
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
            ...     ({'Alice': 1, 'Superman': 1}, 8),
            ...     ({'Alice': 1, 'Terminator': 1}, 9),
            ...     ({'Alice': 1, 'Star Wars': 1}, 8),
            ...     ({'Alice': 1, 'Notting Hill': 1}, 2),
            ...     ({'Alice': 1, 'Harry Potter ': 1}, 5),
            ...     ({'Bob': 1, 'Superman': 1}, 8),
            ...     ({'Bob': 1, 'Terminator': 1}, 9),
            ...     ({'Bob': 1, 'Star Wars': 1}, 8),
            ...     ({'Bob': 1, 'Notting Hill': 1}, 2)
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
            5.320369...

    Note:
        Using a feature scaler such as `preprocessing.StandardScaler` on non-binary features helps
        the optimizer to converge.

    References:
            1. `Factorization Machines <https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf>`_
            2. `Factorization Machines with libFM <https://analyticsconsultores.com.mx/wp-content/uploads/2019/03/Factorization-Machines-with-libFM-Steffen-Rendle-University-of-Konstanz2012-.pdf>`_


    """

    def __init__(self, degree=2, n_factors=10, loss=None, optimizer=None, intercept=0.,
                 intercept_lr=.01, weight_initializer=None, latent_initializer=None, l1=0., l2=0.,
                 clip_gradient=1e12, random_state=None):
        super().__init__(
            degree=degree,
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
        x = self._transform(x)  # One hot encode string modalities
        return self._raw_dot(x)


class FMClassifier(FM, base.BinaryClassifier):
    """Factorization Machines Classifier.

    Parameters:
        degree (int): Polynomial degree or model order.
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
            ...     ({'Alice': 1, 'Superman': 1}, True),
            ...     ({'Alice': 1, 'Terminator': 1}, True),
            ...     ({'Alice': 1, 'Star Wars': 1}, True),
            ...     ({'Alice': 1, 'Notting Hill': 1}, False),
            ...     ({'Alice': 1, 'Harry Potter ': 1}, True),
            ...     ({'Bob': 1, 'Superman': 1}, True),
            ...     ({'Bob': 1, 'Terminator': 1}, True),
            ...     ({'Bob': 1, 'Star Wars': 1}, True),
            ...     ({'Bob': 1, 'Notting Hill': 1}, False)
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
        Using a feature scaler such as `preprocessing.StandardScaler` on non-binary features helps
        the optimizer to converge.

    References:
            1. `Factorization Machines <https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf>`_
            2. `Factorization Machines with libFM <https://analyticsconsultores.com.mx/wp-content/uploads/2019/03/Factorization-Machines-with-libFM-Steffen-Rendle-University-of-Konstanz2012-.pdf>`_

    """

    def __init__(self, degree=2, n_factors=10, loss=None, optimizer=None, intercept=0.,
                 intercept_lr=.01, weight_initializer=None, latent_initializer=None, l1=0., l2=0.,
                 clip_gradient=1e12, random_state=None):
        super().__init__(
            degree=degree,
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
        x = self._transform(x)  # One hot encode string modalities
        p = utils.math.sigmoid(self._raw_dot(x))  # Convert logit to probability
        return {False: 1. - p, True: p}
