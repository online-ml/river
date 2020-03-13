import abc
import collections
import numbers

from .. import optim
from .. import stats
from .. import utils


class BaseFM:
    """Factorization Machines base class.

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

    """

    def __init__(self, n_factors, weight_optimizer, latent_optimizer, loss, sample_normalization,
                 l1_weight, l2_weight, l1_latent, l2_latent, intercept, intercept_lr,
                 weight_initializer, latent_initializer, clip_gradient, seed):
        self.n_factors = n_factors
        self.weight_optimizer = optim.SGD(0.01) if weight_optimizer is None else weight_optimizer
        self.latent_optimizer = optim.SGD(0.01) if latent_optimizer is None else latent_optimizer
        self.loss = loss
        self.sample_normalization = sample_normalization
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
        self.weights = collections.defaultdict(weight_initializer)

        if latent_initializer is None:
            latent_initializer = optim.initializers.Normal(sigma=.1, seed=seed)
        self.latent_initializer = latent_initializer
        self.latents = self._init_latents()

        self.clip_gradient = clip_gradient
        self.seed = seed

    @abc.abstractmethod
    def _init_latents(self) -> collections.defaultdict:
        """Initializes latent weights dict."""

    def fit_one(self, x, y, sample_weight=1.):
        x = self._ohe_cat_features(x)

        if self.sample_normalization:
            x_l2_norm = sum((xj ** 2 for xj in x.values())) ** 0.5
            x = {j: xj / x_l2_norm for j, xj in x.items()}

        return self._fit_one(x, y, sample_weight=sample_weight)

    def _ohe_cat_features(self, x):
        """One hot encodes string features considering them as categorical."""
        return dict((f'{j}_{xj}', 1) if isinstance(xj, str) else (j, xj) for j, xj in x.items())

    def _fit_one(self, x, y, sample_weight=1.):

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
        weights_gradient = self._calculate_weights_gradients(x, g_loss)
        self.weights = self.weight_optimizer.update_after_pred(w=self.weights, g=weights_gradient)

        # Update the latent weights
        self._update_latents(x, g_loss)

    def _raw_dot(self, x):

        # Start with the intercept
        intercept = self.intercept
        y_pred = intercept.get() if isinstance(intercept, stats.Univariate) else intercept

        # Add the unary interactions
        y_pred += utils.math.dot(x, self.weights)

        # Add greater than unary interactions
        y_pred += self._calculate_interactions(x)

        return y_pred

    def _field(self, j):
        """Infers feature field name."""
        return j.split('_')[0]

    @abc.abstractmethod
    def _calculate_interactions(self, x: dict) -> float:
        """Calculates greater than unary interactions."""

    @abc.abstractmethod
    def _calculate_weights_gradients(self, x: dict, g_loss: float) -> dict:
        """Calculates weights gradient."""

    @abc.abstractmethod
    def _update_latents(self, x: dict, g_loss: float):
        """Updates latent weights."""
