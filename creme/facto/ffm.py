import collections
import functools
import itertools
import numpy as np

from .. import base
from .. import optim
from .. import utils

from .base import BaseFM


__all__ = [
    'FFMClassifier',
    'FFMRegressor'
]


class FFM(BaseFM):
    """Field-aware Factorization Machines.

    Parameters:
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

    def __init__(self, n_factors, weight_optimizer, latent_optimizer, loss, sample_normalization,
                 l1_weight, l2_weight, l1_latent, l2_latent, intercept, intercept_lr,
                 weight_initializer, latent_initializer, clip_gradient, seed):
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

    def _init_latents(self):
        random_latents = functools.partial(
            self.latent_initializer,
            shape=self.n_factors
        )
        field_latents_dict = functools.partial(
            collections.defaultdict, random_latents
        )
        return collections.defaultdict(field_latents_dict)

    def _calculate_interactions(self, x):
        """Calculates pairwise interactions."""
        field = self._field
        return sum(
            x[j1] * x[j2] * np.dot(self.latents[j1][field(j2)], self.latents[j2][field(j1)])
            for j1, j2 in itertools.combinations(x.keys(), 2)
        )

    def _calculate_weights_gradients(self, x, g_loss):

        # For notational convenience
        w, l1, l2, sign = self.weights, self.l1_weight, self.l2_weight, utils.math.sign

        return {
            j: g_loss * xj + l1 * sign(w[j]) + l2 * w[j]
            for j, xj in x.items()
        }

    def _update_latents(self, x, g_loss):

        # For notational convenience
        v, l1, l2 = self.latents, self.l1_latent, self.l2_latent
        sign, field = utils.math.sign, self._field

        # Calculate each latent factor gradient before updating any
        latent_gradient = collections.defaultdict(
            lambda: collections.defaultdict(
                lambda: collections.defaultdict(float)
            )
        )

        for j1, j2 in itertools.combinations(x.keys(), 2):
            xj1_xj2 = x[j1] * x[j2]
            field_j1, field_j2 = field(j1), field(j2)

            for f in range(self.n_factors):
                latent_gradient[j1][field_j2][f] += xj1_xj2 * v[j2][field_j1][f]
                latent_gradient[j2][field_j1][f] += xj1_xj2 * v[j1][field_j2][f]

        # Finally update the latent weights
        for j in x.keys():
            for field in latent_gradient[j].keys():
                self.latents[j][field] = self.latent_optimizer.update_after_pred(
                    w=v[j][field],
                    g={
                        f: g_loss * latent_gradient[j][field][f] \
                           + l1 * sign(v[j][field][f]) + 2. * l2 * v[j][field][f]
                        for f in range(self.n_factors)
                    }
                )


class FFMRegressor(FFM, base.Regressor):
    """Field-aware Factorization Machines Regressor.

    Parameters:
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

            >>> model = facto.FFMRegressor(
            ...     n_factors=10,
            ...     intercept=5,
            ...     seed=42,
            ... )

            >>> for x, y in X_y:
            ...     _ = model.fit_one(x, y)

            >>> model.predict_one({'user': 'Bob', 'item': 'Harry Potter', 'time': .14})
            5.319945

    Note:
        - For more efficiency, FM models automatically one hot encode string values considering them as categorical variables (e.g. ``x = {'user': 'Joe'}`` becomes ``x = {'user_Joe': 1}``).
        - Fields are infered from feature names by taking everything before first underscore: ``feature_name.split('_')[0]``.
        - For model stability and better accuracy, numerical features should often be transformed into categorical ones.

    References:
        1. `Juan, Y., Zhuang, Y., Chin, W.S. and Lin, C.J., 2016, September. Field-aware factorization machines for CTR prediction. In Proceedings of the 10th ACM Conference on Recommender Systems (pp. 43-50). <https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf>`_

    """

    def __init__(self, n_factors=10, weight_optimizer=None, latent_optimizer=None, loss=None,
                 sample_normalization=False, l1_weight=0., l2_weight=0., l1_latent=0.,
                 l2_latent=0., intercept=0., intercept_lr=.01, weight_initializer=None,
                 latent_initializer=None, clip_gradient=1e12, seed=None):
        super().__init__(
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


class FFMClassifier(FFM, base.BinaryClassifier):
    """Field-aware Factorization Machines Classifier.

    Parameters:
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
        random_state (int, ``numpy.random.RandomState`` instance or None): If int, ``random_state``
            is the seed used by the random number generator; if ``RandomState`` instance,
            ``random_state`` is the random number generator; if ``None``, the random number
            generator is the ``RandomState`` instance used by `numpy.random`.

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

            >>> model = facto.FFMClassifier(
            ...     n_factors=10,
            ...     intercept=.5,
            ...     seed=42,
            ... )

            >>> for x, y in X_y:
            ...     _ = model.fit_one(x, y)

            >>> model.predict_one({'user': 'Bob', 'item': 'Harry Potter', 'time': .14})
            True

    Note:
        - For more efficiency, FM models automatically one hot encode string values considering them as categorical variables (e.g. ``x = {'user': 'Joe'}`` becomes ``x = {'user_Joe': 1}``).
        - Fields are infered from feature names by taking everything before first underscore: ``feature_name.split('_')[0]``.
        - For model stability and better accuracy, numerical features should often be transformed into categorical ones.

    References:
        1. `Juan, Y., Zhuang, Y., Chin, W.S. and Lin, C.J., 2016, September. Field-aware factorization machines for CTR prediction. In Proceedings of the 10th ACM Conference on Recommender Systems (pp. 43-50). <https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf>`_

    """

    def __init__(self, n_factors=10, weight_optimizer=None, latent_optimizer=None, loss=None,
                 sample_normalization=False, l1_weight=0., l2_weight=0., l1_latent=0.,
                 l2_latent=0., intercept=0., intercept_lr=.01, weight_initializer=None,
                 latent_initializer=None, clip_gradient=1e12, seed=None):
        super().__init__(
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
