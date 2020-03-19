import collections
import functools
import itertools

from .. import base
from .. import optim
from .. import utils

from .base import BaseFM


__all__ = [
    'HOFMClassifier',
    'HOFMRegressor'
]


class HOFM(BaseFM):
    """Higher-Order Factorization Machines.

    Parameters:
        degree (int): Polynomial degree or model order.
        n_factors (int): Dimensionality of the factorization or number of latent factors.
        weight_optimizer (optim.Optimizer): The sequential optimizer used for updating the feature
            weights. Note that the intercept is handled separately.
        latent_optimizer (optim.Optimizer): The sequential optimizer used for updating the latent
            factors.
        loss (optim.Loss): The loss function to optimize for.
        sample_normalization (bool): Whether to divide each element of ``x`` by ``x`` L2-norm.
            Defaults to False.
        l1_weight (float): Amount of L1 regularization used to push weights towards 0.
        l2_weight (float): Amount of L2 regularization used to push weights towards 0.
        l1_latent (float): Amount of L1 regularization used to push latent weights towards 0.
        l2_latent (float): Amount of L2 regularization used to push latent weights towards 0.
        intercept (float or `stats.Univariate` instance): Initial intercept value.
        intercept_lr (optim.schedulers.Scheduler or float): Learning rate scheduler used for
            updating the intercept. If a `float` is passed, then an instance of
            `optim.schedulers.Constant` will be used. Setting this to 0 implies that the intercept
            will be not be updated.
        weight_initializer (optim.initializers.Initializer): Weights initialization scheme.
        latent_initializer (optim.initializers.Initializer): Latent factors initialization scheme.
        clip_gradient (float): Clips the absolute value of each gradient value.
        seed (int): Randomization seed used for reproducibility.

    Attributes:
        weights (collections.defaultdict): The current weights assigned to the features.
        latents (collections.defaultdict): The current latent weights assigned to the features.

    """

    def __init__(self, degree, n_factors, weight_optimizer, latent_optimizer, loss,
                 sample_normalization, l1_weight, l2_weight, l1_latent, l2_latent, intercept,
                 intercept_lr, weight_initializer, latent_initializer, clip_gradient, seed):
        super().__init__(
            n_factors=n_factors,
            weight_optimizer=weight_optimizer,
            latent_optimizer=latent_optimizer,
            loss=loss,
            sample_normalization=sample_normalization,
            l1_weight=l1_weight,
            l2_weight=l2_weight,
            l1_latent=l1_latent,
            l2_latent=l2_latent,
            intercept=intercept,
            intercept_lr=intercept_lr,
            weight_initializer=weight_initializer,
            latent_initializer=latent_initializer,
            clip_gradient=clip_gradient,
            seed=seed
        )
        self.degree = degree

    def _init_latents(self):
        random_latents = functools.partial(
            self.latent_initializer,
            shape=self.n_factors
        )
        order_latents_dict = functools.partial(
            collections.defaultdict, random_latents
        )
        return collections.defaultdict(order_latents_dict)

    def _calculate_interactions(self, x):
        """Calculates greater than unary interactions."""
        return sum(
            self._calculate_interaction(x, l, combination)
            for l in range(2, self.degree + 1)
            for combination in itertools.combinations(x.keys(), l)
        )

    def _calculate_interaction(self, x, l, combination):
        feature_product = functools.reduce(lambda x, y: x * y, (x[j] for j in combination))
        latent_scalar_product = sum(
            functools.reduce(lambda x, y: x * y, (self.latents[j][l][f] for j in combination))
            for f in range(self.n_factors)
        )
        return feature_product * latent_scalar_product

    def _calculate_weights_gradients(self, x, g_loss):

        # For notational convenience
        w, l1, l2, sign = self.weights, self.l1_weight, self.l2_weight, utils.math.sign

        return {
            j: g_loss * xj + l1 * sign(w[j]) + l2 * w[j]
            for j, xj in x.items()
        }

    def _update_latents(self, x, g_loss):

        # For notational convenience
        v, l1, l2, sign = self.latents, self.l1_latent, self.l2_latent, utils.math.sign

        # Calculate each latent factor gradient before updating any
        gradients = collections.defaultdict(
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
                        gradients[j][l][f] += feature_product * latent_product / v[j][l][f]

        # Finally update the latent weights
        for j in x.keys():
            for l in range(2, self.degree + 1):
                self.latents[j][l] = self.latent_optimizer.update_after_pred(
                    w=v[j][l],
                    g={
                        f: g_loss * gradients[j][l][f] + l1 * sign(v[j][l][f]) + 2 * l2 * v[j][l][f]
                        for f in range(self.n_factors)
                    }
                )


class HOFMRegressor(HOFM, base.Regressor):
    """Higher-Order Factorization Machines Regressor.

    Parameters:
        degree (int): Polynomial degree or model order.
        n_factors (int): Dimensionality of the factorization or number of latent factors.
        weight_optimizer (optim.Optimizer): The sequential optimizer used for updating the feature
            weights. Note that the intercept is handled separately.
        latent_optimizer (optim.Optimizer): The sequential optimizer used for updating the latent
            factors.
        loss (optim.Loss): The loss function to optimize for.
        sample_normalization (bool): Whether to divide each element of ``x`` by ``x`` L2-norm.
            Defaults to False.
        l1_weight (float): Amount of L1 regularization used to push weights towards 0.
        l2_weight (float): Amount of L2 regularization used to push weights towards 0.
        l1_latent (float): Amount of L1 regularization used to push latent weights towards 0.
        l2_latent (float): Amount of L2 regularization used to push latent weights towards 0.
        intercept (float or `stats.Univariate` instance): Initial intercept value.
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
        seed (int): Randomization seed used for reproducibility.

    Attributes:
        weights (collections.defaultdict): The current weights assigned to the features.
        latents (collections.defaultdict): The current latent weights assigned to the features.

    Example:

        ::

            >>> from creme import facto

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

            >>> model = facto.HOFMRegressor(
            ...     degree=3,
            ...     n_factors=10,
            ...     intercept=5,
            ...     seed=42,
            ... )

            >>> for x, y in X_y:
            ...     _ = model.fit_one(x, y)

            >>> model.predict_one({'user': 'Bob', 'item': 'Harry Potter', 'time': .14})
            5.311745

    Note:
        - For more efficiency, FM models automatically one hot encode string values considering them as categorical variables.
        - For model stability and better accuracy, numerical features should often be transformed into categorical ones.

    References:
        1. `Rendle, S., 2010, December. Factorization machines. In 2010 IEEE International Conference on Data Mining (pp. 995-1000). IEEE. <https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf>`_

    """

    def __init__(self, degree=3, n_factors=10, weight_optimizer=None, latent_optimizer=None,
                 loss=None, sample_normalization=False, l1_weight=0., l2_weight=0., l1_latent=0.,
                 l2_latent=0., intercept=0., intercept_lr=.01, weight_initializer=None,
                 latent_initializer=None, clip_gradient=1e12, seed=None):
        super().__init__(
            degree=degree,
            n_factors=n_factors,
            weight_optimizer=weight_optimizer,
            latent_optimizer=latent_optimizer,
            loss=optim.losses.Squared() if loss is None else loss,
            sample_normalization=sample_normalization,
            l1_weight=l1_weight,
            l2_weight=l2_weight,
            l1_latent=l1_latent,
            l2_latent=l2_latent,
            intercept=intercept,
            intercept_lr=intercept_lr,
            weight_initializer=weight_initializer,
            latent_initializer=latent_initializer,
            clip_gradient=clip_gradient,
            seed=seed
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
        sample_normalization (bool): Whether to divide each element of ``x`` by ``x`` L2-norm.
            Defaults to False.
        l1_weight (float): Amount of L1 regularization used to push weights towards 0.
        l2_weight (float): Amount of L2 regularization used to push weights towards 0.
        l1_latent (float): Amount of L1 regularization used to push latent weights towards 0.
        l2_latent (float): Amount of L2 regularization used to push latent weights towards 0.
        intercept (float or `stats.Univariate` instance): Initial intercept value.
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
        seed (int): Randomization seed used for reproducibility.

    Attributes:
        weights (collections.defaultdict): The current weights assigned to the features.
        latents (collections.defaultdict): The current latent weights assigned to the features.

    Example:

        ::

            >>> from creme import facto

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

            >>> model = facto.HOFMClassifier(
            ...     degree=3,
            ...     n_factors=10,
            ...     intercept=.5,
            ...     seed=42,
            ... )

            >>> for x, y in X_y:
            ...     _ = model.fit_one(x, y)

            >>> model.predict_one({'user': 'Bob', 'item': 'Harry Potter', 'time': .14})
            True

    Note:
        - For more efficiency, FM models automatically one hot encode string values considering them as categorical variables.
        - For model stability and better accuracy, numerical features should often be transformed into categorical ones.

    References:
        1. `Rendle, S., 2010, December. Factorization machines. In 2010 IEEE International Conference on Data Mining (pp. 995-1000). IEEE. <https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf>`_

    """

    def __init__(self, degree=3, n_factors=10, weight_optimizer=None, latent_optimizer=None,
                 loss=None, sample_normalization=False, l1_weight=0., l2_weight=0., l1_latent=0.,
                 l2_latent=0., intercept=0., intercept_lr=.01, weight_initializer=None,
                 latent_initializer=None, clip_gradient=1e12, seed=None):
        super().__init__(
            degree=degree,
            n_factors=n_factors,
            weight_optimizer=weight_optimizer,
            latent_optimizer=latent_optimizer,
            loss=optim.losses.Log() if loss is None else loss,
            sample_normalization=sample_normalization,
            l1_weight=l1_weight,
            l2_weight=l2_weight,
            l1_latent=l1_latent,
            l2_latent=l2_latent,
            intercept=intercept,
            intercept_lr=intercept_lr,
            weight_initializer=weight_initializer,
            latent_initializer=latent_initializer,
            clip_gradient=clip_gradient,
            seed=seed
        )

    def predict_proba_one(self, x):
        x = self._ohe_cat_features(x)
        p = utils.math.sigmoid(self._raw_dot(x))  # Convert logit to probability
        return {False: 1. - p, True: p}
