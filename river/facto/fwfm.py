import collections
import functools
import itertools
import typing

import numpy as np

from river import base, optim, utils

from .base import BaseFM

__all__ = ["FwFMClassifier", "FwFMRegressor"]


class FwFM(BaseFM):
    """Field-Weighted Factorization Machine base class."""

    def __init__(
        self,
        n_factors,
        weight_optimizer,
        latent_optimizer,
        int_weight_optimizer,
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
        if int_weight_optimizer is None:
            self.int_weight_optimizer = optim.SGD(0.01)
        else:
            self.int_weight_optimizer = int_weight_optimizer

        one = functools.partial(float, 1)
        self.interaction_weights = collections.defaultdict(one)

    def _init_latents(self):
        random_latents = functools.partial(
            self.latent_initializer, shape=self.n_factors
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

        return {j: g_loss * xj + l1 * sign(w[j]) + l2 * w[j] for j, xj in x.items()}

    def _update_latents(
        self, x, g_loss
    ):  # also updates interaction weights as both updates depends of each other

        # For notational convenience
        v, w_int, field = self.latents, self.interaction_weights, self._field
        l1, l2, sign = self.l1_latent, self.l2_latent, utils.math.sign

        # Precompute feature independent sum for time efficiency
        precomputed_sum = {
            f"{j1}_{f}": sum(
                v[j2][f] * xj2 * w_int[field(j1) + field(j2)] for j2, xj2 in x.items()
            )
            for j1, xj1 in x.items()
            for f in range(self.n_factors)
        }

        # Calculate each latent and interaction weights gradients before updating any of them
        latent_gradients = {}
        for j, xj in x.items():
            latent_gradients[j] = {
                f: g_loss
                * (
                    xj * precomputed_sum[f"{j}_{f}"]
                    - v[j][f] * xj * w_int[field(j) + field(j)] ** 2
                )
                + l1 * sign(v[j][f])
                + l2 * v[j][f]
                for f in range(self.n_factors)
            }

        int_gradients = {
            field(j1) + field(j2): g_loss * (x[j1] * x[j2] * np.dot(v[j1], v[j2]))
            for j1, j2 in itertools.combinations(x.keys(), 2)
        }

        # Finally update the latent and interaction weights
        for j in x.keys():
            self.latents[j] = self.latent_optimizer.step(w=v[j], g=latent_gradients[j])

        self.int_weights = self.int_weight_optimizer.step(w=w_int, g=int_gradients)


class FwFMRegressor(FwFM, base.Regressor):
    """Field-weighted Factorization Machine for regression.

    The model equation is defined as:

    $$\\hat{y}(x) = w_{0} + \\sum_{j=1}^{p} w_{j} x_{j}  + \\sum_{j=1}^{p} \\sum_{j'=j+1}^{p} r_{f_j, f_{j'}} \\langle \\mathbf{v}_j, \\mathbf{v}_{j'} \\rangle x_{j} x_{j'}$$

    Where $f_j$ and $f_{j'}$ are $j$ and $j'$ fields, respectively, and $\\mathbf{v}_j$ and
    $\\mathbf{v}_{j'}$ are $j$ and $j'$ latent vectors, respectively.

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
    interaction_weights
        The current interaction strengths of field pairs.

    Examples
    --------

    >>> from river import facto

    >>> dataset = (
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

    >>> for x, y in dataset:
    ...     model = model.learn_one(x, y)

    >>> model.predict_one({'Bob': 1, 'Harry Potter': 1})
    5.236501

    References
    ----------
    [^1]: [Junwei Pan, Jian Xu, Alfonso Lobos Ruiz, Wenliang Zhao, Shengjun Pan, Yu Sun, and Quan Lu, 2018, April. Field-weighted Factorization Machines for Click-Through Rate Prediction in Display Advertising. In Proceedings of the 2018 World Wide Web Conference on World Wide Web. International World Wide Web Conferences Steering Committee, (pp. 1349–1357).](https://arxiv.org/abs/1806.03514)

    """

    def __init__(
        self,
        n_factors=10,
        weight_optimizer: optim.Optimizer = None,
        latent_optimizer: optim.Optimizer = None,
        int_weight_optimizer: optim.Optimizer = None,
        loss: optim.losses.RegressionLoss = None,
        sample_normalization=False,
        l1_weight=0.0,
        l2_weight=0.0,
        l1_latent=0.0,
        l2_latent=0.0,
        intercept=0.0,
        intercept_lr: typing.Union[optim.schedulers.Scheduler, float] = 0.01,
        weight_initializer: optim.initializers.Initializer = None,
        latent_initializer: optim.initializers.Initializer = None,
        clip_gradient=1e12,
        seed: int = None,
    ):

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
            seed=seed,
        )

    def predict_one(self, x):
        x = self._ohe_cat_features(x)
        return self._raw_dot(x)


class FwFMClassifier(FwFM, base.Classifier):
    """Field-weighted Factorization Machine for binary classification.

    The model equation is defined as:

    $$\\hat{y}(x) = w_{0} + \\sum_{j=1}^{p} w_{j} x_{j}  + \\sum_{j=1}^{p} \\sum_{j'=j+1}^{p} r_{f_j, f_{j'}} \\langle \\mathbf{v}_j, \\mathbf{v}_{j'} \\rangle x_{j} x_{j'}$$

    Where $f_j$ and $f_{j'}$ are $j$ and $j'$ fields, respectively, and $\\mathbf{v}_j$ and
    $\\mathbf{v}_{j'}$ are $j$ and $j'$ latent vectors, respectively.

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
    interaction_weights
        The current interaction strengths of field pairs.

    Examples
    --------

    >>> from river import facto

    >>> dataset = (
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

    >>> for x, y in dataset:
    ...     model = model.learn_one(x, y)

    >>> model.predict_one({'Bob': 1, 'Harry Potter': 1})
    True

    References
    ----------
    [^1]: [Junwei Pan, Jian Xu, Alfonso Lobos Ruiz, Wenliang Zhao, Shengjun Pan, Yu Sun, and Quan Lu, 2018, April. Field-weighted Factorization Machines for Click-Through Rate Prediction in Display Advertising. In Proceedings of the 2018 World Wide Web Conference on World Wide Web. International World Wide Web Conferences Steering Committee, (pp. 1349–1357).](https://arxiv.org/abs/1806.03514)

    """

    def __init__(
        self,
        n_factors=10,
        weight_optimizer: optim.Optimizer = None,
        latent_optimizer: optim.Optimizer = None,
        int_weight_optimizer: optim.Optimizer = None,
        loss: optim.losses.BinaryLoss = None,
        sample_normalization=False,
        l1_weight=0.0,
        l2_weight=0.0,
        l1_latent=0.0,
        l2_latent=0.0,
        intercept=0.0,
        intercept_lr: typing.Union[optim.schedulers.Scheduler, float] = 0.01,
        weight_initializer: optim.initializers.Initializer = None,
        latent_initializer: optim.initializers.Initializer = None,
        clip_gradient=1e12,
        seed: int = None,
    ):

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
            seed=seed,
        )

    def predict_proba_one(self, x):
        x = self._ohe_cat_features(x)
        p = utils.math.sigmoid(self._raw_dot(x))  # Convert logit to probability
        return {False: 1.0 - p, True: p}
