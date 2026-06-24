from __future__ import annotations

import collections
import functools
import itertools

import numpy as np

from river import base, optim, utils

from .base import BaseFM

__all__ = ["FFMClassifier", "FFMRegressor"]


class FFM(BaseFM):
    """Field-aware Factorization Machine base class."""

    def __init__(
        self,
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

    def _init_latents(self):
        random_latents = functools.partial(self.latent_initializer, shape=self.n_factors)
        field_latents_dict = functools.partial(collections.defaultdict, random_latents)
        return collections.defaultdict(field_latents_dict)

    @classmethod
    def _unit_test_params(cls):
        yield {"seed": 42}

    def _unit_test_skips(self):
        # Latent factors are initialised lazily, the first time each feature is encountered. Their
        # values therefore depend on the order in which features are first seen, so reshuffling a
        # sample's features yields a different (yet equally valid) model.
        return {"check_shuffle_features_no_impact"}

    def _interaction_names(self, x):
        return [
            f"{j1}({self._field(j2)}) - {j2}({self._field(j1)})"
            for j1, j2 in itertools.combinations(x.keys(), 2)
        ]

    def _interaction_combination_keys(self, x):
        return itertools.combinations(x.keys(), 2)

    def _interaction_val(self, x, combination):
        return functools.reduce(lambda x, y: x * y, (x[j] for j in combination))

    def _interaction_coefficient(self, combination):
        j1, j2 = combination
        return np.dot(self.latents[j1][self._field(j2)], self.latents[j2][self._field(j1)])

    def _calculate_weights_gradients(self, x, g_loss):
        # For notational convenience
        w, l1, l2, sign = self.weights, self.l1_weight, self.l2_weight, utils.math.sign

        return {j: g_loss * xj + l1 * sign(w[j]) + l2 * w[j] for j, xj in x.items()}

    def _update_latents(self, x, g_loss):
        # For notational convenience
        v, l1, l2 = self.latents, self.l1_latent, self.l2_latent
        field = self._field

        # Accumulate each (feature, field) latent gradient as a vector over the factors. NumPy
        # handles the per-factor arithmetic in one shot rather than looping in Python.
        latent_gradient: collections.defaultdict = collections.defaultdict(
            lambda: collections.defaultdict(lambda: np.zeros(self.n_factors))
        )

        for j1, j2 in itertools.combinations(x.keys(), 2):
            xj1_xj2 = x[j1] * x[j2]
            field_j1, field_j2 = field(j1), field(j2)
            latent_gradient[j1][field_j2] += xj1_xj2 * v[j2][field_j1]
            latent_gradient[j2][field_j1] += xj1_xj2 * v[j1][field_j2]

        # Finally update the latent weights
        for j in x.keys():
            for field_, gradient in latent_gradient[j].items():
                vjf = v[j][field_]
                gradient = g_loss * gradient + l1 * np.sign(vjf) + 2.0 * l2 * vjf
                self.latents[j][field_] = self.latent_optimizer.step(
                    w=vjf, g=dict(enumerate(gradient))
                )


class FFMRegressor(FFM, base.Regressor):
    """Field-aware Factorization Machine for regression.

    The model equation is defined by:

    $$\\hat{y}(x) = w_{0} + \\sum_{j=1}^{p} w_{j} x_{j}  + \\sum_{j=1}^{p} \\sum_{j'=j+1}^{p} \\langle \\mathbf{v}_{j, f_{j'}}, \\mathbf{v}_{j', f_j} \\rangle x_{j} x_{j'}$$

    Where $\\mathbf{v}_{j, f_{j'}}$ is the latent vector corresponding to $j$ feature for $f_{j'}$
    field, and $\\mathbf{v}_{j', f_j}$ is the latent vector corresponding to $j'$ feature for $f_j$
    field.

    For more efficiency, this model automatically one-hot encodes strings features considering them
    as categorical variables. Field names are inferred from feature names by taking everything
    before the first underscore: `feature_name.split('_')[0]`.

    Parameters
    ----------
    n_factors
        Dimensionality of the factorization or number of latent factors.
    weight_optimizer
        The sequential optimizer used for updating the feature weights. Note that the intercept is
        handled separately.
    latent_optimizer
        The sequential optimizer used for updating the latent factors.
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

    >>> model = facto.FFMRegressor(
    ...     n_factors=10,
    ...     intercept=5,
    ...     seed=42,
    ... )

    >>> for x, y in dataset:
    ...     model.learn_one(x, y)

    >>> model.predict_one({'user': 'Bob', 'item': 'Harry Potter', 'time': .14})
    5.319945

    >>> report = model.debug_one({'user': 'Bob', 'item': 'Harry Potter', 'time': .14})

    >>> print(report)
    Name                                       Value      Weight     Contribution
                                   Intercept    1.00000    5.23501        5.23501
                                    user_Bob    1.00000    0.11438        0.11438
                                        time    0.14000    0.03186        0.00446
        item_Harry Potter(time) - time(item)    0.14000    0.03153        0.00441
                 user_Bob(time) - time(user)    0.14000    0.02864        0.00401
                           item_Harry Potter    1.00000    0.00000        0.00000
    user_Bob(item) - item_Harry Potter(user)    1.00000   -0.04232       -0.04232

    References
    ----------
    [^1]: [Juan, Y., Zhuang, Y., Chin, W.S. and Lin, C.J., 2016, September. Field-aware factorization machines for CTR prediction. In Proceedings of the 10th ACM Conference on Recommender Systems (pp. 43-50).](https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf)

    """

    def __init__(
        self,
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
        return self._raw_dot(x).item()


class FFMClassifier(FFM, base.Classifier):
    """Field-aware Factorization Machine for binary classification.

    The model equation is defined by:

    $$\\hat{y}(x) = w_{0} + \\sum_{j=1}^{p} w_{j} x_{j}  + \\sum_{j=1}^{p} \\sum_{j'=j+1}^{p} \\langle \\mathbf{v}_{j, f_{j'}}, \\mathbf{v}_{j', f_j} \\rangle x_{j} x_{j'}$$

    Where $\\mathbf{v}_{j, f_{j'}}$ is the latent vector corresponding to $j$ feature for $f_{j'}$
    field, and $\\mathbf{v}_{j', f_j}$ is the latent vector corresponding to $j'$ feature for $f_j$
    field.

    For more efficiency, this model automatically one-hot encodes strings features considering them
    as categorical variables. Field names are inferred from feature names by taking everything
    before the first underscore: `feature_name.split('_')[0]`.

    Parameters
    ----------
    n_factors
        Dimensionality of the factorization or number of latent factors.
    weight_optimizer
        The sequential optimizer used for updating the feature weights. Note that the intercept is
        handled separately.
    latent_optimizer
        The sequential optimizer used for updating the latent factors.
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

    >>> model = facto.FFMClassifier(
    ...     n_factors=10,
    ...     intercept=.5,
    ...     seed=42,
    ... )

    >>> for x, y in dataset:
    ...     model.learn_one(x, y)

    >>> model.predict_one({'user': 'Bob', 'item': 'Harry Potter', 'time': .14})
    True

    References
    ----------
    [^1]: [Juan, Y., Zhuang, Y., Chin, W.S. and Lin, C.J., 2016, September. Field-aware factorization machines for CTR prediction. In Proceedings of the 10th ACM Conference on Recommender Systems (pp. 43-50).](https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf)

    """

    def __init__(
        self,
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
