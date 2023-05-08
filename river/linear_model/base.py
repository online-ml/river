from __future__ import annotations

import contextlib
import numbers

import numpy as np
import pandas as pd

from river import optim, utils

__all__ = ["GLM"]


class GLM:
    """Generalized Linear Model.

    This serves as a base class for linear and logistic regression.

    Parameters
    ----------
    optimizer
        The sequential optimizer used for updating the weights. Note that the intercept updates are
        handled separately.
    loss
        The loss function to optimize for.
    l2
        Amount of L2 regularization used to push weights towards 0.
        For now, only one type of penalty can be used. The joint use of L1 and L2 is not explicitly supported.
    l1
        Amount of L1 regularization used to push weights towards 0.
        For now, only one type of penalty can be used. The joint use of L1 and L2 is not explicitly supported.
    intercept_init
        Initial intercept value.
    intercept_lr
        Learning rate scheduler used for updating the intercept. A `optim.schedulers.Constant` is
        used if a `float` is provided. The intercept is not updated when this is set to 0.
    clip_gradient
        Clips the absolute value of each gradient value.
    initializer
        Weights initialization scheme.

    """

    def __init__(
        self,
        optimizer,
        loss,
        l2,
        l1,
        intercept_init,
        intercept_lr,
        clip_gradient,
        initializer,
    ):
        self.optimizer = optimizer
        self.loss = loss
        self.l2 = l2
        self.l1 = l1
        self.intercept_init = intercept_init
        self.intercept = intercept_init
        self.intercept_lr = (
            optim.schedulers.Constant(intercept_lr)
            if isinstance(intercept_lr, numbers.Number)
            else intercept_lr
        )
        self.clip_gradient = clip_gradient
        self.initializer = initializer
        self._weights = utils.VectorDict(None)

        # The predict_many functions are going to return pandas.Series. We can name the series with
        # the name given to the y series seen during the last learn_many call.
        self._y_name = None

        if l1 != 0:
            if l2 != 0:
                raise NotImplementedError(
                    "The joint use of L1 and L2 penalties is currently not explicitly supported!"
                )

            # L1-specific parameters
            self.max_cum_l1 = 0
            self.cum_l1 = utils.VectorDict(None, optim.initializers.Zeros())

    @property
    def _mutable_attributes(self):
        return {"optimizer", "l2", "l1", "loss", "intercept_lr", "clip_gradient", "initializer"}

    @property
    def weights(self):
        return self._weights.to_dict()

    @contextlib.contextmanager
    def _learn_mode(self, mask=None):
        weights = self._weights
        try:
            # enable the initializer and set a mask
            self._weights = utils.VectorDict(weights, self.initializer, mask)
            yield
        finally:
            self._weights = weights

    def _get_intercept_update(self, loss_gradient):
        return self.intercept_lr.get(self.optimizer.n_iterations) * loss_gradient

    def _fit(self, x, y, w, get_grad):
        # Some optimizers need to do something before a prediction is made
        self.optimizer.look_ahead(w=self._weights)

        # Calculate the gradient
        gradient, loss_gradient = get_grad(x, y, w)

        # Update the intercept
        self.intercept -= self._get_intercept_update(loss_gradient)

        # Update the weights
        self.optimizer.step(w=self._weights, g=gradient)

        # Apply L1 cumulative penalty if applicable
        if self.l1 != 0.0:
            # This should be called after the learning_rate update in case of adaptive learning rate
            self.max_cum_l1 = self.max_cum_l1 + self.l1 * self.optimizer.learning_rate

            self._update_weights(x)

        return self

    def _update_weights(self, x):
        # L1 cumulative penalty helper

        # Apply penalty to each weight iteratively, with the potential of being parrallelized by using VectorDict
        for j, xj in x.items():
            wj_temp = self._weights[j]

            if wj_temp > 0:
                self._weights[j] = max(0, wj_temp - (self.max_cum_l1 + self.cum_l1[j]))
            elif wj_temp < 0:
                self._weights[j] = min(0, wj_temp + (self.max_cum_l1 - self.cum_l1[j]))
            else:
                self._weights[j] = wj_temp

            # Update the penalty state of the estimator
            self.cum_l1[j] = self.cum_l1[j] + (self._weights[j] - wj_temp)

    # Single instance methods

    def _raw_dot_one(self, x: dict) -> float:
        return self._weights @ utils.VectorDict(x) + self.intercept

    def _eval_gradient_one(self, x: dict, y: float, w: float) -> tuple[dict, float]:
        loss_gradient = self.loss.gradient(y_true=y, y_pred=self._raw_dot_one(x))
        loss_gradient *= w
        loss_gradient = float(
            utils.math.clamp(loss_gradient, -self.clip_gradient, self.clip_gradient)
        )

        if self.l2:
            return (
                loss_gradient * utils.VectorDict(x) + self.l2 * self._weights,
                loss_gradient,
            )

        return (loss_gradient * utils.VectorDict(x), loss_gradient)

    def learn_one(self, x, y, w=1.0):
        with self._learn_mode(x):
            return self._fit(x, y, w, get_grad=self._eval_gradient_one)

    # Mini-batch methods

    def _raw_dot_many(self, X: pd.DataFrame) -> np.ndarray:
        return X.values @ self._weights.to_numpy(X.columns) + self.intercept

    def _eval_gradient_many(
        self, X: pd.DataFrame, y: pd.Series, w: float | pd.Series
    ) -> tuple[dict, float]:
        loss_gradient = self.loss.gradient(y_true=y.values, y_pred=self._raw_dot_many(X))
        loss_gradient *= w
        loss_gradient = np.clip(loss_gradient, -self.clip_gradient, self.clip_gradient)

        # At this point we have a feature matrix X of shape (n, p). The loss gradient is a vector
        # of length p. We want to multiply each of X's rows by the corresponding value in the loss
        # gradient. When this is all done, we collapse X by computing the average of each column,
        # thereby obtaining the mean gradient of the batch. From thereon, the code reduces to the
        # single instance case.
        gradient = np.einsum("ij,i->ij", X.values, loss_gradient).mean(axis=0)
        if self.l2:
            gradient += self.l2 * self._weights.to_numpy(X.columns)

        return dict(zip(X.columns, gradient)), loss_gradient.mean()

    def learn_many(self, X: pd.DataFrame, y: pd.Series, w: float | pd.Series = 1):
        self._y_name = y.name
        with self._learn_mode(set(X)):
            return self._fit(X, y, w, get_grad=self._eval_gradient_many)
