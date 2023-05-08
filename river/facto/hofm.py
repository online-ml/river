from __future__ import annotations

import collections
import functools
import itertools

import numpy as np

from river import base, optim, utils

from .base import BaseFM

__all__ = ["HOFMClassifier", "HOFMRegressor"]


class HOFM(BaseFM):
    """Higher-Order Factorization Machine base class."""

    def __init__(
        self,
        degree,
        n_factors,
        weight_optimizer,
        latent_optimizer,
        loss,
        sample_normalization,
        l1_weight,
        l2_weight,
        l1_latent,
        l2_latent,
        intercept,
        intercept_lr,
        weight_initializer,
        latent_initializer,
        clip_gradient,
        seed,
    ):
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
            seed=seed,
        )
        self.degree = degree

    def _init_latents(self):
        random_latents = functools.partial(self.latent_initializer, shape=self.n_factors)
        order_latents_dict = functools.partial(collections.defaultdict, random_latents)
        return collections.defaultdict(order_latents_dict)

    def _interaction_names(self, x):
        return [
            " - ".join(map(str, combination))
            for d in range(2, self.degree + 1)
            for combination in itertools.combinations(x.keys(), d)
        ]

    def _interaction_combination_keys(self, x):
        for d in range(2, self.degree + 1):
            yield from itertools.combinations(x.keys(), d)

    def _interaction_val(self, x, combination):
        return functools.reduce(lambda x, y: x * y, (x[j] for j in combination))

    def _interaction_coefficient(self, combination):
        return sum(
            functools.reduce(
                lambda x, y: np.multiply(x, y),
                (self.latents[j][len(combination)] for j in combination),
            )
        )

    def _calculate_weights_gradients(self, x, g_loss):
        # For notational convenience
        w, l1, l2, sign = self.weights, self.l1_weight, self.l2_weight, utils.math.sign

        return {j: g_loss * xj + l1 * sign(w[j]) + l2 * w[j] for j, xj in x.items()}

    def _update_latents(self, x, g_loss):
        # For notational convenience
        v, l1, l2, sign = self.latents, self.l1_latent, self.l2_latent, utils.math.sign

        # Calculate each latent factor gradient before updating any
        gradients = collections.defaultdict(
            lambda: collections.defaultdict(lambda: collections.defaultdict(float))
        )

        for d in range(2, self.degree + 1):
            for combination in itertools.combinations(x.keys(), d):
                feature_product = functools.reduce(lambda x, y: x * y, (x[j] for j in combination))

                for f in range(self.n_factors):
                    latent_product = functools.reduce(
                        lambda x, y: x * y, (v[j][d][f] for j in combination)
                    )

                    for j in combination:
                        gradients[j][d][f] += feature_product * latent_product / v[j][d][f]

        # Finally update the latent weights
        for j in x.keys():
            for d in range(2, self.degree + 1):
                self.latents[j][d] = self.latent_optimizer.step(
                    w=v[j][d],
                    g={
                        f: g_loss * gradients[j][d][f] + l1 * sign(v[j][d][f]) + 2 * l2 * v[j][d][f]
                        for f in range(self.n_factors)
                    },
                )


class HOFMRegressor(HOFM, base.Regressor):
    """Higher-Order Factorization Machine for regression.

    The model equation is defined as:

    $$\\hat{y}(x) = w_{0} + \\sum_{j=1}^{p} w_{j} x_{j}  + \\sum_{l=2}^{d} \\sum_{j_1=1}^{p} \\cdots \\sum_{j_l=j_{l-1}+1}^{p} \\left(\\prod_{j'=1}^{l} x_{j_{j'}} \\right) \\left(\\sum_{f=1}^{k_l} \\prod_{j'=1}^{l} v_{j_{j'}, f}^{(l)} \\right)$$

    For more efficiency, this model automatically one-hot encodes strings features considering
    them as categorical variables.

    Parameters
    ----------
    degree
        Polynomial degree or model order.
    n_factors
        Dimensionality of the factorization or number of latent factors.
    weight_optimizer
        The sequential optimizer used for updating the feature weights. Note thatthe intercept is
        handled separately.
    latent_optimizer
        The sequential optimizer used for updating the latent factors.
    int_weight_optimizer
        The sequential optimizer used for updating the field pairs interaction weights.
    loss
        The loss function to optimize for.
    sample_normalization
        Whether to divide each element of `x` by `x`'s L2-norm.
    l1_weight
        Amount of L1 regularization used to push weights towards 0.
    l2_weight
        Amount of L2 regularization used to push weights towards 0.
    l1_latent
        Amount of L1 regularization used to push latent weights towards 0.
    l2_latent
        Amount of L2 regularization used to push latent weights towards 0.
    intercept
        Initial intercept value.
    intercept_lr
        Learning rate scheduler used for updating the intercept. An instance of
        `optim.schedulers.Constant` is used if a `float` is passed. No intercept will be used
        if this is set to 0.
    weight_initializer
        Weights initialization scheme. Defaults to `optim.initializers.Zeros()`.
    latent_initializer
        Latent factors initialization scheme. Defaults to
        `optim.initializers.Normal(mu=.0, sigma=.1, random_state=self.random_state)`.
    clip_gradient
        Clips the absolute value of each gradient value.
    seed
        Randomization seed used for reproducibility.

    Attributes
    ----------
    weights
        The current weights assigned to the features.
    latents
        The current latent weights assigned to the features.

    Examples
    --------

    >>> from river import facto

    >>> dataset = (
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

    >>> for x, y in dataset:
    ...     _ = model.learn_one(x, y)

    >>> model.predict_one({'user': 'Bob', 'item': 'Harry Potter', 'time': .14})
    5.311745

    >>> report = model.debug_one({'user': 'Bob', 'item': 'Harry Potter', 'time': .14})

    >>> print(report)
    Name                                  Value      Weight     Contribution
                              Intercept    1.00000    5.23495        5.23495
                               user_Bob    1.00000    0.11436        0.11436
                                   time    0.14000    0.03185        0.00446
                        user_Bob - time    0.14000    0.00884        0.00124
    user_Bob - item_Harry Potter - time    0.14000    0.00117        0.00016
                      item_Harry Potter    1.00000    0.00000        0.00000
               item_Harry Potter - time    0.14000   -0.00695       -0.00097
           user_Bob - item_Harry Potter    1.00000   -0.04246       -0.04246

    References
    ----------
    [^1]: [Rendle, S., 2010, December. Factorization machines. In 2010 IEEE International Conference on Data Mining (pp. 995-1000). IEEE.](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)

    """

    def __init__(
        self,
        degree=3,
        n_factors=10,
        weight_optimizer: optim.base.Optimizer | None = None,
        latent_optimizer: optim.base.Optimizer | None = None,
        loss: optim.losses.RegressionLoss | None = None,
        sample_normalization=False,
        l1_weight=0.0,
        l2_weight=0.0,
        l1_latent=0.0,
        l2_latent=0.0,
        intercept=0.0,
        intercept_lr: optim.base.Scheduler | float = 0.01,
        weight_initializer: optim.initializers.Initializer | None = None,
        latent_initializer: optim.initializers.Initializer | None = None,
        clip_gradient=1e12,
        seed: int | None = None,
    ):
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
            seed=seed,
        )

    def predict_one(self, x):
        x = self._ohe_cat_features(x)
        return self._raw_dot(x)


class HOFMClassifier(HOFM, base.Classifier):
    """Higher-Order Factorization Machine for binary classification.

    The model equation is defined as:

    $$\\hat{y}(x) = w_{0} + \\sum_{j=1}^{p} w_{j} x_{j}  + \\sum_{l=2}^{d} \\sum_{j_1=1}^{p} \\cdots \\sum_{j_l=j_{l-1}+1}^{p} \\left(\\prod_{j'=1}^{l} x_{j_{j'}} \\right) \\left(\\sum_{f=1}^{k_l} \\prod_{j'=1}^{l} v_{j_{j'}, f}^{(l)} \\right)$$

    For more efficiency, this model automatically one-hot encodes strings features considering
    them as categorical variables.

    Parameters
    ----------
    degree
        Polynomial degree or model order.
    n_factors
        Dimensionality of the factorization or number of latent factors.
    weight_optimizer
        The sequential optimizer used for updating the feature weights. Note that the intercept is
        handled separately.
    latent_optimizer
        The sequential optimizer used for updating the latent factors.
    int_weight_optimizer
        The sequential optimizer used for updating the field pairs interaction weights.
    loss
        The loss function to optimize for.
    sample_normalization
        Whether to divide each element of `x` by `x`'s L2-norm.
    l1_weight
        Amount of L1 regularization used to push weights towards 0.
    l2_weight
        Amount of L2 regularization used to push weights towards 0.
    l1_latent
        Amount of L1 regularization used to push latent weights towards 0.
    l2_latent
        Amount of L2 regularization used to push latent weights towards 0.
    intercept
        Initial intercept value.
    intercept_lr
        Learning rate scheduler used for updating the intercept. An instance of
        `optim.schedulers.Constant` is used if a `float` is passed. No intercept will be used
        if this is set to 0.
    weight_initializer
        Weights initialization scheme. Defaults to `optim.initializers.Zeros()`.
    latent_initializer
        Latent factors initialization scheme. Defaults to
        `optim.initializers.Normal(mu=.0, sigma=.1, random_state=self.random_state)`.
    clip_gradient
        Clips the absolute value of each gradient value.
    seed
        Randomization seed used for reproducibility.

    Attributes
    ----------
    weights
        The current weights assigned to the features.
    latents
        The current latent weights assigned to the features.

    Examples
    --------

    >>> from river import facto

    >>> dataset = (
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

    >>> for x, y in dataset:
    ...     _ = model.learn_one(x, y)

    >>> model.predict_one({'user': 'Bob', 'item': 'Harry Potter', 'time': .14})
    True

    References
    ----------
    [^1]: [Rendle, S., 2010, December. Factorization machines. In 2010 IEEE International Conference on Data Mining (pp. 995-1000). IEEE.](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)

    """

    def __init__(
        self,
        degree=3,
        n_factors=10,
        weight_optimizer: optim.base.Optimizer | None = None,
        latent_optimizer: optim.base.Optimizer | None = None,
        loss: optim.losses.BinaryLoss | None = None,
        sample_normalization=False,
        l1_weight=0.0,
        l2_weight=0.0,
        l1_latent=0.0,
        l2_latent=0.0,
        intercept=0.0,
        intercept_lr: optim.base.Scheduler | float = 0.01,
        weight_initializer: optim.initializers.Initializer | None = None,
        latent_initializer: optim.initializers.Initializer | None = None,
        clip_gradient=1e12,
        seed: int | None = None,
    ):
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
            seed=seed,
        )

    def predict_proba_one(self, x):
        x = self._ohe_cat_features(x)
        p = utils.math.sigmoid(self._raw_dot(x))  # Convert logit to probability
        return {False: 1.0 - p, True: p}
