import collections
import copy
import functools
import typing

import numpy as np

from river import optim
from river import utils

from . import base


__all__ = ["FunkMF"]


class FunkMF(base.Recommender):
    """Funk Matrix Factorization for recommender systems.

    The model equation is defined as:

    $$\\hat{y}(x) = \\langle \\mathbf{v}_u, \\mathbf{v}_i \\rangle = \\sum_{f=1}^{k} \\mathbf{v}_{u, f} \\cdot \\mathbf{v}_{i, f}$$

    where $k$ is the number of latent factors.

    This model expects a dict input with a `user` and an `item` entries without any type
    constraint on their values (i.e. can be strings or numbers). Other entries are ignored.

    Parameters
    ----------
    n_factors
        Dimensionality of the factorization or number of latent factors.
    optimizer
        The sequential optimizer used for updating the latent factors.
    loss
        The loss function to optimize for.
    l2
        Amount of L2 regularization used to push weights towards 0.
    initializer
        Latent factors initialization scheme.
    clip_gradient
        Clips the absolute value of each gradient value.
    seed
        Randomization seed used for reproducibility.

    Attributes
    ----------
    u_latents : collections.defaultdict
        The user latent vectors randomly initialized.
    i_latents : collections.defaultdict
        The item latent vectors randomly initialized.
    u_optimizer : optim.Optimizer
        The sequential optimizer used for updating the user latent weights.
    i_optimizer : optim.Optimizer
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

    >>> model = reco.FunkMF(
    ...     n_factors=10,
    ...     optimizer=optim.SGD(0.1),
    ...     initializer=optim.initializers.Normal(mu=0., sigma=0.1, seed=11),
    ... )

    >>> for x, y in dataset:
    ...     _ = model.learn_one(x, y)

    >>> model.predict_one({'user': 'Bob', 'item': 'Harry Potter'})
    1.866272

    References
    ----------
    [^1]: [Netflix update: Try this at home](https://sifter.org/simon/journal/20061211.html)
    [^2]: [Matrix factorization techniques for recommender systems](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)

    """

    def __init__(
        self,
        n_factors=10,
        optimizer: optim.Optimizer = None,
        loss: optim.losses.Loss = None,
        l2=0.0,
        initializer: optim.initializers.Initializer = None,
        clip_gradient=1e12,
        seed: int = None,
    ):

        self.n_factors = n_factors
        self.u_optimizer = optim.SGD() if optimizer is None else copy.deepcopy(optimizer)
        self.i_optimizer = optim.SGD() if optimizer is None else copy.deepcopy(optimizer)
        self.loss = optim.losses.Squared() if loss is None else loss
        self.l2 = l2

        if initializer is None:
            initializer = optim.initializers.Normal(mu=0.0, sigma=0.1, seed=seed)
        self.initializer = initializer

        self.clip_gradient = clip_gradient
        self.seed = seed

        random_latents = functools.partial(self.initializer, shape=self.n_factors)
        self.u_latents: typing.DefaultDict[
            int, optim.initializers.Initializer
        ] = collections.defaultdict(random_latents)
        self.i_latents: typing.DefaultDict[
            int, optim.initializers.Initializer
        ] = collections.defaultdict(random_latents)

    def _predict_one(self, user, item):
        return np.dot(self.u_latents[user], self.i_latents[item])

    def _learn_one(self, user, item, y):

        # Calculate the gradient of the loss with respect to the prediction
        g_loss = self.loss.gradient(y, self._predict_one(user, item))

        # Clamp the gradient to avoid numerical instability
        g_loss = utils.math.clamp(g_loss, minimum=-self.clip_gradient, maximum=self.clip_gradient)

        # Calculate latent gradients
        u_latent_grad = {user: g_loss * self.i_latents[item] + self.l2 * self.u_latents[user]}
        i_latent_grad = {item: g_loss * self.u_latents[user] + self.l2 * self.i_latents[item]}

        # Update latent weights
        self.u_latents = self.u_optimizer.update_after_pred(self.u_latents, u_latent_grad)
        self.i_latents = self.i_optimizer.update_after_pred(self.i_latents, i_latent_grad)

        return self
