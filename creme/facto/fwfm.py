import collections
import functools
import itertools
import numpy as np

from .. import base
from .. import optim
from .. import utils

from .base import BaseFM


__all__ = [
    'FwFMClassifier',
    'FwFMRegressor'
]


class FwFM(BaseFM):
    """Field-Weighted Factorization Machines.

    Parameters:
        n_factors (int): Dimensionality of the factorization or number of latent factors.
        weight_optimizer (optim.Optimizer): The sequential optimizer used for updating the feature
            weights. Note that the intercept is handled separately.
        latent_optimizer (optim.Optimizer): The sequential optimizer used for updating the latent
            factors.
        int_weight_optimizer (optim.Optimizer): The sequential optimizer used for updating the field
            pairs interaction weights.
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
        interaction_weights (collections.defaultdict): The current interaction strengths of field
            pairs.

    """

    def __init__(self, n_factors, weight_optimizer, latent_optimizer, int_weight_optimizer, loss,
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
        if int_weight_optimizer is None:
            self.int_weight_optimizer = optim.SGD(0.01)
        else:
            self.int_weight_optimizer = int_weight_optimizer

        one = functools.partial(float, 1)
        self.interaction_weights = collections.defaultdict(one)

    def _init_latents(self):
        random_latents = functools.partial(
            self.latent_initializer,
            shape=self.n_factors
        )
        return collections.defaultdict(random_latents)

    def _calculate_interactions(self, x):
        """Calculates pairwise interactions."""

        # For notational convenience
        v, w_int, field = self.latents, self.interaction_weights, self._field

        return sum(
            x[j1] * x[j2] * np.dot(v[j1], v[j2]) * w_int[field(j1) + field(j2)]
            for j1, j2 in itertools.combinations(x.keys(), 2)
        )

    def _calculate_weights_gradients(self, x, g_loss):

        # For notational convenience
        w, l1, l2, sign = self.weights, self.l1_weight, self.l2_weight, utils.math.sign

        return {
            j: g_loss * xj + l1 * sign(w[j]) + l2 * w[j]
            for j, xj in x.items()
        }

    def _update_latents(self, x, g_loss): # also updates interaction weights as both updates depends of each other

        # For notational convenience
        v, w_int, field = self.latents, self.interaction_weights, self._field
        l1, l2, sign = self.l1_latent, self.l2_latent, utils.math.sign

        # Precompute feature independent sum for time efficiency
        precomputed_sum = {
            f'{j1}_{f}': sum(
                v[j2][f] * xj2 * w_int[field(j1) + field(j2)]
                for j2, xj2 in x.items()
            )
            for j1, xj1 in x.items()
            for f in range(self.n_factors)
        }

        # Calculate each latent and interaction weights gradients before updating any of them
        latent_gradients = {}
        for j, xj in x.items():
            latent_gradients[j] = {
                f: g_loss * (xj * precomputed_sum[f'{j}_{f}'] - v[j][f] * xj \
                * w_int[field(j) + field(j)] ** 2) + l1 * sign(v[j][f]) + l2 * v[j][f]
                for f in range(self.n_factors)
            }

        int_gradients = {
            field(j1) + field(j2): g_loss * (x[j1] * x[j2] * np.dot(v[j1], v[j2]))
            for j1, j2 in itertools.combinations(x.keys(), 2)
        }

        # Finally update the latent and interaction weights
        for j in x.keys():
            self.latents[j] = self.latent_optimizer.update_after_pred(w=v[j], g=latent_gradients[j])

        self.int_weights = self.int_weight_optimizer.update_after_pred(w=w_int, g=int_gradients)


class FwFMRegressor(FwFM, base.Regressor):
    """Field-weighted Factorization Machines Regressor.

    Parameters:
        n_factors (int): Dimensionality of the factorization or number of latent factors.
        weight_optimizer (optim.Optimizer): The sequential optimizer used for updating the feature
            weights. Note that the intercept is handled separately.
        latent_optimizer (optim.Optimizer): The sequential optimizer used for updating the latent
            factors.
        int_weight_optimizer (optim.Optimizer): The sequential optimizer used for updating the field
            pairs interaction weights.
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
        interaction_weights (collections.defaultdict): The current interaction strengths of field
            pairs.

    Example:

        ::

            >>> from creme import facto

            >>> X_y = (
            ...     ({'user': 'Alice', 'item': 'Superman'}, 8),
            ...     ({'user': 'Alice', 'item': 'Terminator'}, 9),
            ...     ({'user': 'Alice', 'item': 'Star Wars'}, 8),
            ...     ({'user': 'Alice', 'item': 'Notting Hill'}, 2),
            ...     ({'user': 'Alice', 'item': 'Harry Potter '}, 5),
            ...     ({'user': 'Bob', 'item': 'Superman'}, 8),
            ...     ({'user': 'Bob', 'item': 'Terminator'}, 9),
            ...     ({'user': 'Bob', 'item': 'Star Wars'}, 8),
            ...     ({'user': 'Bob', 'item': 'Notting Hill'}, 2)
            ... )

            >>> model = facto.FwFMRegressor(
            ...     n_factors=10,
            ...     intercept=5,
            ...     seed=42,
            ... )

            >>> for x, y in X_y:
            ...     _ = model.fit_one(x, y)

            >>> model.predict_one({'Bob': 1, 'Harry Potter': 1})
            5.236501

    Note:
        - For more efficiency, FM models automatically one hot encode string values considering them as categorical variables.
        - For model stability and better accuracy, numerical features should often be transformed into categorical ones.

    References:
        1. `Junwei Pan, Jian Xu, Alfonso Lobos Ruiz, Wenliang Zhao, Shengjun Pan, Yu Sun, and Quan Lu, 2018, April. Field-weighted Factorization Machines for Click-Through Rate Prediction in Display Advertising. In Proceedings of the 2018 World Wide Web Conference on World Wide Web. International World Wide Web Conferences Steering Committee, (pp. 1349–1357). <https://arxiv.org/abs/1806.03514>`_


    """

    def __init__(self, n_factors=10, weight_optimizer=None, latent_optimizer=None,
                 int_weight_optimizer=None, loss=None, sample_normalization=False, l1_weight=0.,
                 l2_weight=0., l1_latent=0., l2_latent=0., intercept=0., intercept_lr=.01,
                 weight_initializer=None, latent_initializer=None, clip_gradient=1e12,
                 seed=None):
        super().__init__(
            n_factors=n_factors,
            weight_optimizer=weight_optimizer,
            int_weight_optimizer=int_weight_optimizer,
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


class FwFMClassifier(FwFM, base.BinaryClassifier):
    """Field-weighted Factorization Machines Classifier.

    Parameters:
        n_factors (int): Dimensionality of the factorization or number of latent factors.
        weight_optimizer (optim.Optimizer): The sequential optimizer used for updating the feature
            weights. Note that the intercept is handled separately.
        latent_optimizer (optim.Optimizer): The sequential optimizer used for updating the latent
            factors.
        int_weight_optimizer (optim.Optimizer): The sequential optimizer used for updating the field
            pairs interaction weights.
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
        interaction_weights (collections.defaultdict): The current interaction strengths of field
            pairs.

    Example:

        ::

            >>> from creme import facto

            >>> X_y = (
            ...     ({'user': 'Alice', 'item': 'Superman'}, True),
            ...     ({'user': 'Alice', 'item': 'Terminator'}, True),
            ...     ({'user': 'Alice', 'item': 'Star Wars'}, True),
            ...     ({'user': 'Alice', 'item': 'Notting Hill'}, False),
            ...     ({'user': 'Alice', 'item': 'Harry Potter '}, True),
            ...     ({'user': 'Bob', 'item': 'Superman'}, True),
            ...     ({'user': 'Bob', 'item': 'Terminator'}, True),
            ...     ({'user': 'Bob', 'item': 'Star Wars'}, True),
            ...     ({'user': 'Bob', 'item': 'Notting Hill'}, False)
            ... )

            >>> model = facto.FwFMClassifier(
            ...     n_factors=10,
            ...     seed=42,
            ... )

            >>> for x, y in X_y:
            ...     _ = model.fit_one(x, y)

            >>> model.predict_one({'Bob': 1, 'Harry Potter': 1})
            True

    Note:
        - For more efficiency, FM models automatically one hot encode string values considering them as categorical variables.
        - For model stability and better accuracy, numerical features should often be transformed into categorical ones.

    References:
        1. `Junwei Pan, Jian Xu, Alfonso Lobos Ruiz, Wenliang Zhao, Shengjun Pan, Yu Sun, and Quan Lu, 2018, April. Field-weighted Factorization Machines for Click-Through Rate Prediction in Display Advertising. In Proceedings of the 2018 World Wide Web Conference on World Wide Web. International World Wide Web Conferences Steering Committee, (pp. 1349–1357). <https://arxiv.org/abs/1806.03514>`_

    """

    def __init__(self, n_factors=10, weight_optimizer=None, latent_optimizer=None,
                 int_weight_optimizer=None, loss=None, sample_normalization=False, l1_weight=0.,
                 l2_weight=0., l1_latent=0., l2_latent=0., intercept=0., intercept_lr=.01,
                 weight_initializer=None, latent_initializer=None, clip_gradient=1e12,
                 seed=None):
        super().__init__(
            n_factors=n_factors,
            weight_optimizer=weight_optimizer,
            int_weight_optimizer=int_weight_optimizer,
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
