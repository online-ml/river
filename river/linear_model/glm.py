import contextlib
import numbers
import typing

import numpy as np
import pandas as pd

from river import optim, utils


class GLM:
    """Generalized Linear Model.

    This serves as a base class for linear and logistic regression.

    """

    def __init__(
        self,
        optimizer,
        loss,
        l2,
        intercept_init,
        intercept_lr,
        clip_gradient,
        initializer,
    ):
        self.optimizer = optimizer
        self.loss = loss
        self.l2 = l2
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

        return self

    # Single instance methods

    def _raw_dot_one(self, x: dict) -> float:
        return self._weights @ utils.VectorDict(x) + self.intercept

    def _eval_gradient_one(self, x: dict, y: float, w: float) -> (dict, float):

        loss_gradient = self.loss.gradient(y_true=y, y_pred=self._raw_dot_one(x))
        loss_gradient *= w
        loss_gradient = float(
            utils.math.clamp(loss_gradient, -self.clip_gradient, self.clip_gradient)
        )

        return (
            loss_gradient * utils.VectorDict(x) + self.l2 * self._weights,
            loss_gradient,
        )

    def learn_one(self, x, y, w=1.0):
        with self._learn_mode(x):
            return self._fit(x, y, w, get_grad=self._eval_gradient_one)

    # Mini-batch methods

    def _raw_dot_many(self, X: pd.DataFrame) -> np.ndarray:
        return X.values @ self._weights.to_numpy(X.columns) + self.intercept

    def _eval_gradient_many(
        self, X: pd.DataFrame, y: pd.Series, w: typing.Union[float, pd.Series]
    ) -> (dict, float):

        loss_gradient = self.loss.gradient(
            y_true=y.values, y_pred=self._raw_dot_many(X)
        )
        loss_gradient *= w
        loss_gradient = np.clip(loss_gradient, -self.clip_gradient, self.clip_gradient)

        # At this point we have a feature matrix X of shape (n, p). The loss gradient is a vector
        # of length p. We want to multiply each of X's rows by the corresponding value in the loss
        # gradient. When this is all done, we collapse X by computing the average of each column,
        # thereby obtaining the mean gradient of the batch. From thereon, the code reduces to the
        # single instance case.
        gradient = np.einsum("ij,i->ij", X.values, loss_gradient).mean(axis=0)

        return dict(zip(X.columns, gradient)), loss_gradient.mean()

    def learn_many(
        self, X: pd.DataFrame, y: pd.Series, w: typing.Union[float, pd.Series] = 1
    ):
        self._y_name = y.name
        with self._learn_mode(set(X)):
            return self._fit(X, y, w, get_grad=self._eval_gradient_many)
