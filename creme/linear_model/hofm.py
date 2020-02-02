import collections
import functools
import itertools
import numbers

from .. import base
from .. import optim
from .. import stats
from .. import utils


__all__ = [
    'HOFMClassifier',
    'HOFMRegressor'
]


class HOFM:
    """Higher-Order Factorization Machines.

    Parameters:
        degree (int): Polynomial degree or model order.
        n_factors (int): Dimensionality of the factorization or number of latent factors.
        weight_optimizer (optim.Optimizer): The sequential optimizer used for updating the feature
            weights. Note that the intercept is handled separately.
        latent_optimizer (optim.Optimizer): The sequential optimizer used for updating the latent
            factors.
        loss (optim.Loss): The loss function to optimize for.
        l1_weight (float): Amount of L1 regularization used to push weights towards 0.
        l2_weight (float): Amount of L2 regularization used to push weights towards 0.
        l1_latent (float): Amount of L1 regularization used to push latent weights towards 0.
        l2_latent (float): Amount of L2 regularization used to push latent weights towards 0.
        intercept (float or `creme.stats.Univariate` instance): Initial intercept value.
        intercept_lr (optim.schedulers.Scheduler or float): Learning rate scheduler used for
            updating the intercept. If a `float` is passed, then an instance of
            `optim.schedulers.Constant` will be used. Setting this to 0 implies that the intercept
            will be not be updated.
        weight_initializer (optim.initializers.Initializer): Weights initialization scheme. Defaults
            to ``optim.initializers.Zeros()``.
        latent_initializer (optim.initializers.Initializer): Latent factors initialization scheme.
            Defaults to
            ``optim.initializers.Normal(mu=.0, sigma=.1, random_state=self.random_state)``.
        clip_gradient (float): Clips the absolute value of each gradient value.
        random_state (int, ``numpy.random.RandomState`` instance or None): If int, ``random_state``
            is the seed used by the random number generator; if ``RandomState`` instance,
            ``random_state`` is the random number generator; if ``None``, the random number
            generator is the ``RandomState`` instance used by `numpy.random`.

    Attributes:
        weights (collections.defaultdict): The current weights assigned to the features.
        latents (collections.defaultdict): The current latent weights assigned to the features.

    """

    def __init__(self, degree, n_factors, weight_optimizer, latent_optimizer, loss, l1_weight,
                 l2_weight, l1_latent, l2_latent, intercept, intercept_lr, weight_initializer,
                 latent_initializer, clip_gradient, random_state):
        self.degree = degree
        self.n_factors = n_factors
        self.weight_optimizer = optim.SGD(0.01) if weight_optimizer is None else weight_optimizer
        self.latent_optimizer = optim.SGD(0.01) if latent_optimizer is None else latent_optimizer
        self.loss = loss
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.l1_latent = l1_latent
        self.l2_latent = l2_latent
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

        self.clip_gradient = clip_gradient
        self.random_state = random_state

        random_latents = functools.partial(
            latent_initializer,
            shape=n_factors
        )

        order_latents_dict = functools.partial(
            collections.defaultdict, random_latents
        )

        self.weights = collections.defaultdict(weight_initializer)
        self.latents = collections.defaultdict(order_latents_dict)

    def _ohe_cat_features(self, x):
        """One hot encodes string features considering them as categorical."""
        return dict((f'{j}_{xj}', 1) if isinstance(xj, str) else (j, xj) for j, xj in x.items())

    def fit_one(self, x, y, sample_weight=1.):
        x = self._ohe_cat_features(x)
        return self._fit_one(x, y, sample_weight=sample_weight)

    def _calculate_interaction(self, x, l, combination):
        feature_product = functools.reduce(lambda x, y: x * y, (x[j] for j in combination))
        latent_scalar_product = sum(
            functools.reduce(lambda x, y: x * y, (self.latents[j][l][f] for j in combination))
            for f in range(self.n_factors)
        )
        return feature_product * latent_scalar_product

    def _raw_dot(self, x):

        # Start with the intercept
        intercept = self.intercept
        y_pred = intercept.get() if isinstance(intercept, stats.Univariate) else intercept

        # Add the unary interactions
        y_pred += utils.math.dot(x, self.weights)

        # Add greater than unary interactions
        y_pred += sum(
            self._calculate_interaction(x, l, combination)
            for l in range(2, self.degree + 1)
            for combination in itertools.combinations(x.keys(), l)
        )

        return y_pred

    def _sign(self, x):
        return -1 if x < 0 else (1 if x > 0 else 0)

    def _fit_one(self, x, y, sample_weight=1.):

        # For notational convenience
        w, v = self.weights, self.latents
        k, sign = self.n_factors, self._sign
        l1_weight, l2_weight = self.l1_weight, self.l2_weight
        l1_latent, l2_latent = self.l1_latent, self.l2_latent

        # Update the intercept if statistic before calculating the gradient
        if isinstance(self.intercept, stats.Univariate):
            self.intercept.update(y)

        # Calculate the gradient of the loss with respect to the raw output
        g_loss = self.loss.gradient(y_true=y, y_pred=self._raw_dot(x))

        # Clamp the gradient to avoid numerical instability
        g_loss = utils.math.clamp(g_loss, minimum=-self.clip_gradient, maximum=self.clip_gradient)

        # Apply the sample weight
        g_loss *= sample_weight

        # Update the intercept if not statistic
        if not isinstance(self.intercept, stats.Univariate):
            w0_lr = self.intercept_lr.get(self.weight_optimizer.n_iterations)
            self.intercept -= w0_lr * g_loss

        # Update the weights
        weight_gradient = {
            j: g_loss * xj + l1_weight * sign(w[j]) + l2_weight * w[j]
            for j, xj in x.items()
        }
        self.weights = self.weight_optimizer.update_after_pred(w=w, g=weight_gradient)

        # Update the latent weights
        latent_gradient = collections.defaultdict(
            lambda: collections.defaultdict(
                lambda: collections.defaultdict(float)
            )
        )

        for l in range(2, self.degree + 1):
            for combination in itertools.combinations(x.keys(), l):
                feature_product = functools.reduce(lambda x, y: x * y, (x[j] for j in combination))

                for f in range(self.n_factors):
                    latent_product = functools.reduce(lambda x, y: x * y, (v[j][l][f] for j in combination))

                    for j in combination:
                        latent_gradient[j][l][f] += feature_product * latent_product / v[j][l][f]

        for j in x.keys():
            for l in range(2, self.degree + 1):
                self.latents[j][l] = self.latent_optimizer.update_after_pred(
                    w=v[j][l],
                    g={
                        f: g_loss * latent_gradient[j][l][f] + l1_latent * sign(v[j][l][f]) \
                           + 2. * l2_latent * v[j][l][f]
                        for f in range(k)
                    }
                )

        return self


class HOFMRegressor(HOFM, base.Regressor):
    """Higer-Order Factorization Machines Regressor.

    Parameters:
        degree (int): Polynomial degree or model order.
        n_factors (int): Dimensionality of the factorization or number of latent factors.
        weight_optimizer (optim.Optimizer): The sequential optimizer used for updating the feature
            weights. Note that the intercept is handled separately.
        latent_optimizer (optim.Optimizer): The sequential optimizer used for updating the latent
            factors.
        loss (optim.Loss): The loss function to optimize for.
        l1_weight (float): Amount of L1 regularization used to push weights towards 0.
        l2_weight (float): Amount of L2 regularization used to push weights towards 0.
        l1_latent (float): Amount of L1 regularization used to push latent weights towards 0.
        l2_latent (float): Amount of L2 regularization used to push latent weights towards 0.
        intercept (float or `creme.stats.Univariate` instance): Initial intercept value.
        intercept_lr (optim.schedulers.Scheduler or float): Learning rate scheduler used for
            updating the intercept. If a `float` is passed, then an instance of
            `optim.schedulers.Constant` will be used. Setting this to 0 implies that the intercept
            will be not be updated.
        weight_initializer (optim.initializers.Initializer): Weights initialization scheme. Defaults
            to ``optim.initializers.Zeros()``.
        latent_initializer (optim.initializers.Initializer): Latent factors initialization scheme.
            Defaults to
            ``optim.initializers.Normal(mu=.0, sigma=.1, random_state=self.random_state)``.
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
            ...     ({'user': 'Alice', 'item': 'Superman', 'time': .12}, 8),
            ...     ({'user': 'Alice', 'item': 'Terminator', 'time': .13}, 9),
            ...     ({'user': 'Alice', 'item': 'Star Wars', 'time': .14}, 8),
            ...     ({'user': 'Alice', 'item': 'Notting Hill', 'time': .15}, 2),
            ...     ({'user': 'Alice', 'item': 'Harry Potter ', 'time': .16}, 5),
            ...     ({'user': 'Bob', 'item': 'Superman', 'time': .13}, 8),
            ...     ({'user': 'Bob', 'item': 'Terminator', 'time': .12}, 9),
            ...     ({'user': 'Bob', 'item': 'Star Wars', 'time': .16}, 8),
            ...     ({'user': 'Bob', 'item': 'Notting Hill', 'time': .10}, 2)
            ... )

            >>> model = linear_model.HOFMRegressor(
            ...     degree=3,
            ...     n_factors=10,
            ...     intercept=5,
            ...     random_state=42,
            ... )

            >>> for x, y in X_y:
            ...     _ = model.fit_one(x, y)

            >>> model.predict_one({'user': 'Bob', 'item': 'Harry Potter', 'time': .14})
            5.311745...

    Note:
        - For more efficiency, FM models automatically one hot encode string values considering them as categorical variables.
        - For model stability and better accuracy, numerical features should often be transformed into categorical ones.

    References:
        1. `Factorization Machines <https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf>`_

    """

    def __init__(self, degree=3, n_factors=10, weight_optimizer=None, latent_optimizer=None,
                 loss=None, l1_weight=0., l2_weight=0., l1_latent=0., l2_latent=0., intercept=0.,
                 intercept_lr=.01, weight_initializer=None, latent_initializer=None,
                 clip_gradient=1e12, random_state=None):
        super().__init__(
            degree=degree,
            n_factors=n_factors,
            weight_optimizer=weight_optimizer,
            latent_optimizer=latent_optimizer,
            loss=optim.losses.Squared() if loss is None else loss,
            l1_weight=l1_weight,
            l2_weight=l2_weight,
            l1_latent=l1_latent,
            l2_latent=l2_latent,
            intercept=intercept,
            intercept_lr=intercept_lr,
            weight_initializer=weight_initializer,
            latent_initializer=latent_initializer,
            clip_gradient=clip_gradient,
            random_state=random_state
        )

    def predict_one(self, x):
        x = self._ohe_cat_features(x)
        return self._raw_dot(x)


class HOFMClassifier(HOFM, base.BinaryClassifier):
    """Higher-Order Factorization Machines Classifier.

    Parameters:
        degree (int): Polynomial degree or model order.
        n_factors (int): Dimensionality of the factorization or number of latent factors.
        weight_optimizer (optim.Optimizer): The sequential optimizer used for updating the feature
            weights. Note that the intercept is handled separately.
        latent_optimizer (optim.Optimizer): The sequential optimizer used for updating the latent
            factors.
        loss (optim.Loss): The loss function to optimize for.
        l1_weight (float): Amount of L1 regularization used to push weights towards 0.
        l2_weight (float): Amount of L2 regularization used to push weights towards 0.
        l1_latent (float): Amount of L1 regularization used to push latent weights towards 0.
        l2_latent (float): Amount of L2 regularization used to push latent weights towards 0.
        intercept (float or `creme.stats.Univariate` instance): Initial intercept value.
        intercept_lr (optim.schedulers.Scheduler or float): Learning rate scheduler used for
            updating the intercept. If a `float` is passed, then an instance of
            `optim.schedulers.Constant` will be used. Setting this to 0 implies that the intercept
            will be not be updated.
        weight_initializer (optim.initializers.Initializer): Weights initialization scheme. Defaults
            to ``optim.initializers.Zeros()``.
        latent_initializer (optim.initializers.Initializer): Latent factors initialization scheme.
            Defaults to
            ``optim.initializers.Normal(mu=.0, sigma=.1, random_state=self.random_state)``.
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
            ...     ({'user': 'Alice', 'item': 'Superman', 'time': .12}, True),
            ...     ({'user': 'Alice', 'item': 'Terminator', 'time': .13}, True),
            ...     ({'user': 'Alice', 'item': 'Star Wars', 'time': .14}, True),
            ...     ({'user': 'Alice', 'item': 'Notting Hill', 'time': .15}, False),
            ...     ({'user': 'Alice', 'item': 'Harry Potter ', 'time': .16}, True),
            ...     ({'user': 'Bob', 'item': 'Superman', 'time': .13}, True),
            ...     ({'user': 'Bob', 'item': 'Terminator', 'time': .12}, True),
            ...     ({'user': 'Bob', 'item': 'Star Wars', 'time': .16}, True),
            ...     ({'user': 'Bob', 'item': 'Notting Hill', 'time': .10}, False)
            ... )

            >>> model = linear_model.HOFMClassifier(
            ...     degree=3,
            ...     n_factors=10,
            ...     intercept=.5,
            ...     random_state=42,
            ... )

            >>> for x, y in X_y:
            ...     _ = model.fit_one(x, y)

            >>> model.predict_one({'user': 'Bob', 'item': 'Harry Potter', 'time': .14})
            True

    Note:
        - For more efficiency, FM models automatically one hot encode string values considering them as categorical variables.
        - For model stability and better accuracy, numerical features should often be transformed into categorical ones.

    References:
        1. `Factorization Machines <https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf>`_

    """

    def __init__(self, degree=3, n_factors=10, weight_optimizer=None, latent_optimizer=None,
                 loss=None, l1_weight=0., l2_weight=0., l1_latent=0., l2_latent=0., intercept=0.,
                 intercept_lr=.01, weight_initializer=None, latent_initializer=None,
                 clip_gradient=1e12, random_state=None):
        super().__init__(
            degree=degree,
            n_factors=n_factors,
            weight_optimizer=weight_optimizer,
            latent_optimizer=latent_optimizer,
            loss=optim.losses.Log() if loss is None else loss,
            l1_weight=l1_weight,
            l2_weight=l2_weight,
            l1_latent=l1_latent,
            l2_latent=l2_latent,
            intercept=intercept,
            intercept_lr=intercept_lr,
            weight_initializer=weight_initializer,
            latent_initializer=latent_initializer,
            clip_gradient=clip_gradient,
            random_state=random_state
        )

    def predict_proba_one(self, x):
        x = self._ohe_cat_features(x)
        p = utils.math.sigmoid(self._raw_dot(x))  # Convert logit to probability
        return {False: 1. - p, True: p}
