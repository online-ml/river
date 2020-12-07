import collections
import functools
import itertools
import typing

import numpy as np

from river import base
from river import optim
from river import utils

from .base import BaseFM


__all__ = ["FMClassifier", "FMRegressor"]


class FM(BaseFM):
    """Factorization machine base class."""

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
        return collections.defaultdict(random_latents)

    def _calculate_interactions(self, x):
        """Calculates pairwise interactions."""
        return sum(
            x[j1] * x[j2] * np.dot(self.latents[j1], self.latents[j2])
            for j1, j2 in itertools.combinations(x.keys(), 2)
        )

    def _calculate_weights_gradients(self, x, g_loss):

        # For notational convenience
        w, l1, l2, sign = self.weights, self.l1_weight, self.l2_weight, utils.math.sign

        return {j: g_loss * xj + l1 * sign(w[j]) + l2 * w[j] for j, xj in x.items()}

    def _update_latents(self, x, g_loss):

        # For notational convenience
        v, l1, l2, sign = self.latents, self.l1_latent, self.l2_latent, utils.math.sign

        # Precompute feature independent sum for time efficiency
        precomputed_sum = {
            f: sum(v[j][f] * xj for j, xj in x.items()) for f in range(self.n_factors)
        }

        # Calculate each latent factor gradient before updating any
        gradients = {}
        for j, xj in x.items():
            gradients[j] = {
                f: g_loss * (xj * precomputed_sum[f] - v[j][f] * xj ** 2)
                + l1 * sign(v[j][f])
                + l2 * v[j][f]
                for f in range(self.n_factors)
            }

        # Finally update the latent weights
        for j in x.keys():
            self.latents[j] = self.latent_optimizer.update_after_pred(w=v[j], g=gradients[j])


class FMRegressor(FM, base.Regressor):
    """Factorization Machine for regression.

    The model equation is defined as:

    $$\\hat{y}(x) = w_{0} + \\sum_{j=1}^{p} w_{j} x_{j}  + \\sum_{j=1}^{p} \\sum_{j'=j+1}^{p} \\langle \\mathbf{v}_j, \\mathbf{v}_{j'} \\rangle x_{j} x_{j'}$$

    Where $\\mathbf{v}_j$ and $\\mathbf{v}_{j'}$ are $j$ and $j'$ latent vectors, respectively.

    For more efficiency, this model automatically one-hot encodes strings features considering them
    as categorical variables.

    Parameters
    ----------
    n_factors
        Dimensionality of the factorization or number of latent factors.
    weight_optimizer
        The sequential optimizer used for updating the feature weights. Note that
        the intercept is handled separately.
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

    >>> model = facto.FMRegressor(
    ...     n_factors=10,
    ...     intercept=5,
    ...     seed=42,
    ... )

    >>> for x, y in dataset:
    ...     _ = model.learn_one(x, y)

    >>> model.predict_one({'Bob': 1, 'Harry Potter': 1})
    5.236504

    References
    ----------
    [^1]: [Rendle, S., 2010, December. Factorization machines. In 2010 IEEE International Conference on Data Mining (pp. 995-1000). IEEE.](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
    [^2]: [Rendle, S., 2012, May. Factorization Machines with libFM. In ACM Transactions on Intelligent Systems and Technology 3, 3, Article 57, 22 pages.](https://analyticsconsultores.com.mx/wp-content/uploads/2019/03/Factorization-Machines-with-libFM-Steffen-Rendle-University-of-Konstanz2012-.pdf)

    """

    def __init__(
        self,
        n_factors=10,
        weight_optimizer: optim.Optimizer = None,
        latent_optimizer: optim.Optimizer = None,
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


class FMClassifier(FM, base.Classifier):
    """Factorization Machine for binary classification.

    The model equation is defined as:

    $$\\hat{y}(x) = w_{0} + \\sum_{j=1}^{p} w_{j} x_{j}  + \\sum_{j=1}^{p} \\sum_{j'=j+1}^{p} \\langle \\mathbf{v}_j, \\mathbf{v}_{j'} \\rangle x_{j} x_{j'}$$

    Where $\\mathbf{v}_j$ and $\\mathbf{v}_{j'}$ are $j$ and $j'$ latent vectors, respectively.

    For more efficiency, this model automatically one-hot encodes strings features considering them
    as categorical variables.

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

    >>> model = facto.FMClassifier(
    ...     n_factors=10,
    ...     seed=42,
    ... )

    >>> for x, y in dataset:
    ...     _ = model.learn_one(x, y)

    >>> model.predict_one({'Bob': 1, 'Harry Potter': 1})
    True

    References
    ----------
    [^1]: [Rendle, S., 2010, December. Factorization machines. In 2010 IEEE International Conference on Data Mining (pp. 995-1000). IEEE.](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
    [^2]: [Rendle, S., 2012, May. Factorization Machines with libFM. In ACM Transactions on Intelligent Systems and Technology 3, 3, Article 57, 22 pages.](https://analyticsconsultores.com.mx/wp-content/uploads/2019/03/Factorization-Machines-with-libFM-Steffen-Rendle-University-of-Konstanz2012-.pdf)

    """

    def __init__(
        self,
        n_factors=10,
        weight_optimizer: optim.Optimizer = None,
        latent_optimizer: optim.Optimizer = None,
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
