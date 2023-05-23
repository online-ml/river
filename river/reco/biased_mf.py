from __future__ import annotations

import collections
import copy
import functools

import numpy as np

from river import optim, stats, utils

from .base import Ranker

__all__ = ["BiasedMF"]


class BiasedMF(Ranker):
    """Biased Matrix Factorization for recommender systems.

    The model equation is defined as:

    $$\\hat{y}(x) = \\bar{y} + bu_{u} + bi_{i} + \\langle \\mathbf{v}_u, \\mathbf{v}_i \\rangle$$

    Where $bu_{u}$ and $bi_{i}$ are respectively the user and item biases. The last term being
    simply the dot product between the latent vectors of the given user-item pair:

    $$\\langle \\mathbf{v}_u, \\mathbf{v}_i \\rangle = \\sum_{f=1}^{k} \\mathbf{v}_{u, f} \\cdot \\mathbf{v}_{i, f}$$

    where $k$ is the number of latent factors.

    This model expects a dict input with a `user` and an `item` entries without any type constraint
    on their values (i.e. can be strings or numbers). Other entries are ignored.

    Parameters
    ----------
    n_factors
        Dimensionality of the factorization or number of latent factors.
    bias_optimizer
        The sequential optimizer used for updating the bias weights.
    latent_optimizer
        The sequential optimizer used for updating the latent weights.
    loss
        The loss function to optimize for.
    l2_bias
        Amount of L2 regularization used to push bias weights towards 0.
    l2_latent
        Amount of L2 regularization used to push latent weights towards 0.
    weight_initializer
        Weights initialization scheme.
    latent_initializer
        Latent factors initialization scheme.
    clip_gradient
        Clips the absolute value of each gradient value.
    seed
        Random number generation seed. Set this for reproducibility.

    Attributes
    ----------
    global_mean : stats.Mean
        The target arithmetic mean.
    u_biases : collections.defaultdict
        The user bias weights.
    i_biases : collections.defaultdict
        The item bias weights.
    u_latents : collections.defaultdict
        The user latent vectors randomly initialized.
    i_latents : collections.defaultdict
        The item latent vectors randomly initialized.
    u_bias_optimizer : optim.base.Optimizer
        The sequential optimizer used for updating the user bias weights.
    i_bias_optimizer : optim.base.Optimizer
        The sequential optimizer used for updating the item bias weights.
    u_latent_optimizer : optim.base.Optimizer
        The sequential optimizer used for updating the user latent weights.
    i_latent_optimizer : optim.base.Optimizer
        The sequential optimizer used for updating the item latent weights.

    Examples
    --------

    >>> from river import optim
    >>> from river import reco

    >>> dataset = (
    ...     ({'user': 'Alice', 'item': 'Superman'}, 8),
    ...     ({'user': 'Alice', 'item': 'Terminator'}, 9),
    ...     ({'user': 'Alice', 'item': 'Star Wars'}, 8),
    ...     ({'user': 'Alice', 'item': 'Notting Hill'}, 2),
    ...     ({'user': 'Alice', 'item': 'Harry Potter'}, 5),
    ...     ({'user': 'Bob', 'item': 'Superman'}, 8),
    ...     ({'user': 'Bob', 'item': 'Terminator'}, 9),
    ...     ({'user': 'Bob', 'item': 'Star Wars'}, 8),
    ...     ({'user': 'Bob', 'item': 'Notting Hill'}, 2)
    ... )

    >>> model = reco.BiasedMF(
    ...     n_factors=10,
    ...     bias_optimizer=optim.SGD(0.025),
    ...     latent_optimizer=optim.SGD(0.025),
    ...     latent_initializer=optim.initializers.Normal(mu=0., sigma=0.1, seed=71)
    ... )

    >>> for x, y in dataset:
    ...     _ = model.learn_one(**x, y=y)

    >>> model.predict_one(user='Bob', item='Harry Potter')
    6.489025

    References
    ----------
    [^1]: [Paterek, A., 2007, August. Improving regularized singular value decomposition for collaborative filtering. In Proceedings of KDD cup and workshop (Vol. 2007, pp. 5-8)](https://www.cs.uic.edu/~liub/KDD-cup-2007/proceedings/Regular-Paterek.pdf)
    [^2]: [Matrix factorization techniques for recommender systems](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)

    """

    def __init__(
        self,
        n_factors=10,
        bias_optimizer: optim.base.Optimizer | None = None,
        latent_optimizer: optim.base.Optimizer | None = None,
        loss: optim.losses.Loss | None = None,
        l2_bias=0.0,
        l2_latent=0.0,
        weight_initializer: optim.initializers.Initializer | None = None,
        latent_initializer: optim.initializers.Initializer | None = None,
        clip_gradient=1e12,
        seed=None,
    ):
        super().__init__(seed=seed)

        self.bias_optimizer = bias_optimizer
        self.latent_optimizer = latent_optimizer

        self.n_factors = n_factors
        self.u_bias_optimizer = (
            optim.SGD() if bias_optimizer is None else copy.deepcopy(bias_optimizer)
        )
        self.i_bias_optimizer = (
            optim.SGD() if bias_optimizer is None else copy.deepcopy(bias_optimizer)
        )
        self.u_latent_optimizer = (
            optim.SGD() if latent_optimizer is None else copy.deepcopy(latent_optimizer)
        )
        self.i_latent_optimizer = (
            optim.SGD() if latent_optimizer is None else copy.deepcopy(latent_optimizer)
        )
        self.loss = optim.losses.Squared() if loss is None else loss
        self.l2_bias = l2_bias
        self.l2_latent = l2_latent

        if weight_initializer is None:
            weight_initializer = optim.initializers.Zeros()
        self.weight_initializer = weight_initializer

        if latent_initializer is None:
            latent_initializer = optim.initializers.Normal(sigma=0.1, seed=self.seed)
        self.latent_initializer = latent_initializer

        self.clip_gradient = clip_gradient
        self.global_mean = stats.Mean()

        self.u_biases: collections.defaultdict[
            int, optim.initializers.Initializer
        ] = collections.defaultdict(weight_initializer)
        self.i_biases: collections.defaultdict[
            int, optim.initializers.Initializer
        ] = collections.defaultdict(weight_initializer)

        random_latents = functools.partial(self.latent_initializer, shape=self.n_factors)
        self.u_latents: collections.defaultdict[
            int, optim.initializers.Initializer
        ] = collections.defaultdict(random_latents)
        self.i_latents: collections.defaultdict[
            int, optim.initializers.Initializer
        ] = collections.defaultdict(random_latents)

    @property
    def _mutable_attributes(self):
        return {
            "bias_optimizer",
            "latent_optimizer",
            "loss",
            "l2_bias",
            "l2_latent",
            "weight_initializer",
            "weight_initializer",
            "clip_gradient",
        }

    def predict_one(self, user, item, x=None):
        # Initialize the prediction to the mean
        y_pred = self.global_mean.get()

        # Add the user bias
        y_pred += self.u_biases[user]

        # Add the item bias
        y_pred += self.i_biases[item]

        # Add the dot product of the user and the item latent vectors
        y_pred += np.dot(self.u_latents[user], self.i_latents[item])

        return y_pred

    def learn_one(self, user, item, y, x=None):
        # Update the global mean
        self.global_mean.update(y)

        # Calculate the gradient of the loss with respect to the prediction
        g_loss = self.loss.gradient(y, self.predict_one(user, item))

        # Clamp the gradient to avoid numerical instability
        g_loss = utils.math.clamp(g_loss, minimum=-self.clip_gradient, maximum=self.clip_gradient)

        # Calculate weights gradients
        u_grad_bias = {user: g_loss + self.l2_bias * self.u_biases[user]}
        i_grad_bias = {item: g_loss + self.l2_bias * self.i_biases[item]}
        u_latent_grad = {
            user: g_loss * self.i_latents[item] + self.l2_latent * self.u_latents[user]
        }
        i_latent_grad = {
            item: g_loss * self.u_latents[user] + self.l2_latent * self.i_latents[item]
        }

        # Update weights
        self.u_biases = self.u_bias_optimizer.step(self.u_biases, u_grad_bias)
        self.i_biases = self.i_bias_optimizer.step(self.i_biases, i_grad_bias)
        self.u_latents = self.u_latent_optimizer.step(self.u_latents, u_latent_grad)
        self.i_latents = self.i_latent_optimizer.step(self.i_latents, i_latent_grad)

        return self
