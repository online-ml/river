from __future__ import annotations

import abc
import collections
import numbers
from collections.abc import Iterable

import numpy as np

from river import optim, utils


class BaseFM:
    """Factorization Machines base class."""

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
            if isinstance(intercept_lr, numbers.Number)
            else intercept_lr
        )

        if weight_initializer is None:
            weight_initializer = optim.initializers.Zeros()
        self.weight_initializer = weight_initializer
        self.weights = collections.defaultdict(weight_initializer)

        if latent_initializer is None:
            latent_initializer = optim.initializers.Normal(sigma=0.1, seed=seed)
        self.latent_initializer = latent_initializer
        self.latents = self._init_latents()

        self.clip_gradient = clip_gradient
        self.seed = seed

    @abc.abstractmethod
    def _init_latents(self) -> collections.defaultdict:
        """Initializes latent weights dict."""

    def learn_one(self, x, y, sample_weight=1.0):
        x = self._ohe_cat_features(x)

        if self.sample_normalization:
            x_l2_norm = sum(xj**2 for xj in x.values()) ** 0.5
            x = {j: xj / x_l2_norm for j, xj in x.items()}

        return self._learn_one(x, y, sample_weight=sample_weight)

    def _ohe_cat_features(self, x):
        """One hot encodes string features considering them as categorical."""
        return dict((f"{j}_{xj}", 1) if isinstance(xj, str) else (j, xj) for j, xj in x.items())

    def _learn_one(self, x, y, sample_weight=1.0):
        # Calculate the gradient of the loss with respect to the raw output
        g_loss = self.loss.gradient(y_true=y, y_pred=self._raw_dot(x))

        # Clamp the gradient to avoid numerical instability
        g_loss = utils.math.clamp(g_loss, minimum=-self.clip_gradient, maximum=self.clip_gradient)

        # Apply the sample weight
        g_loss *= sample_weight

        # Update the intercept
        intercept_lr = self.intercept_lr.get(self.weight_optimizer.n_iterations)
        self.intercept -= intercept_lr * g_loss

        # Update the weights
        weights_gradient = self._calculate_weights_gradients(x, g_loss)
        self.weights = self.weight_optimizer.step(w=self.weights, g=weights_gradient)

        # Update the latent weights
        self._update_latents(x, g_loss)

        return self

    def _raw_dot(self, x):
        # Start with the intercept
        y_pred = self.intercept

        # Add the unary interactions
        y_pred += utils.math.dot(x, self.weights)

        # Add greater than unary interactions
        y_pred += self._calculate_interactions(x)

        return y_pred

    def _field(self, j):
        """Infers feature field name."""
        return j.split("_")[0]

    def _calculate_interactions(self, x: dict) -> float:
        """Calculates greater than unary interactions.
        For normal FM: sigma (i < j) product(latents[xi] * latents[xj]) xi * xj
        interaction_combination_keys: sigma (i < j)
        interaction_val: xi * xj
        interaction_coefficient: latents[xi] * latents[xj]
        """
        return sum(
            self._interaction_coefficient(combination) * self._interaction_val(x, combination)
            for combination in self._interaction_combination_keys(x)
        )

    def debug_one(self, x: dict, decimals: int = 5) -> str:
        """Debugs the output of the FM regressor.

        Parameters
        ----------
        x
            A dictionary of features.
        decimals
            The number of decimals use for printing each numeric value.

        Returns
        -------
        A table which explains the output.
        """

        x = self._ohe_cat_features(x)

        def fmt_float(x: float) -> str:
            return "{: ,.{prec}f}".format(x, prec=decimals)

        names = (
            self._interaction_names(x)  # latents
            + list(map(str, x.keys()))  # weights
            + ["Intercept"]  # intercept
        )

        values = list(
            map(
                fmt_float,
                [
                    self._interaction_val(x, combination)
                    for combination in self._interaction_combination_keys(x)
                ]  # latents
                + list(x.values())  # weights
                + [1],  # intercept
            )
        )

        weights = list(
            map(
                fmt_float,
                [
                    self._interaction_coefficient(combination)
                    for combination in self._interaction_combination_keys(x)
                ]  # latents
                + [self.weights.get(i, 0) for i in x]  # weights
                + [self.intercept],  # intercept
            )
        )
        contributions = (
            [
                self._interaction_coefficient(combination) * self._interaction_val(x, combination)
                for combination in self._interaction_combination_keys(x)
            ]  # latents
            + [xi * self.weights.get(i, 0) for i, xi in x.items()]  # weights
            + [self.intercept]  # intercept
        )
        order = list(reversed(np.argsort(contributions)))
        contributions_str = list(map(fmt_float, contributions))

        table = utils.pretty.print_table(
            headers=["Name", "Value", "Weight", "Contribution"],
            columns=[names, values, weights, contributions_str],
            order=order,
        )

        return table

    @abc.abstractmethod
    def _interaction_names(self, x) -> list:
        "Return names for interactions."

    @abc.abstractmethod
    def _interaction_combination_keys(self, x) -> Iterable:
        """Return combinations for interactions."""

    @abc.abstractmethod
    def _interaction_val(self, x, combination) -> float:
        """Return values corresponding to a given combination of interaction."""

    @abc.abstractmethod
    def _interaction_coefficient(self, combination) -> float:
        """Return coefficient corresponding to a given combination of interaction."""

    @abc.abstractmethod
    def _calculate_weights_gradients(self, x: dict, g_loss: float) -> dict:
        """Calculates weights gradient."""

    @abc.abstractmethod
    def _update_latents(self, x: dict, g_loss: float):
        """Updates latent weights."""
