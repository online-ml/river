import contextlib
import numbers
import typing

import numpy as np
import pandas as pd

from river import optim, utils

import collections


class GLM:
    """Generalized Linear Model.

    This serves as a base class for linear and logistic regression.

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

        self.max_cum_l1 = 0
        # self.cum_l1 = collections.defaultdict(float)
        self.cum_l1 = utils.VectorDict(None, optim.initializers.Zeros())

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

        # REFLECTION: this should be stated after the learning_rate update in case of adaptive learning_rate
        self.max_cum_l1 = self.max_cum_l1 + self.l1 * self.optimizer.learning_rate
        # /

        if (self.l1 != 0.0) and (self.l2 == 0.0):
            # print("hello")
            self._update_weights(x)
            # pass

        return self

    # def _update_weights(self, x, loss_gradient):
    def _update_weights(self, x):
        # REFLECTION: learning_rate is a multiplier in optimizers;
        # if it's a multiplier for weight updates here aswell, we could omit it somewhere
        # and have this penalty implemented not inside the optimizer, which would be easier

        for j, xj in x.items():
            # REFLECTION: it's not `temperature` it's `temporary`
            # wj_temp = self.weights[j] + self.learning_rate * xj * loss_gradient
            wj_temp = self._weights[j]
            self._apply_penalty(j, wj_temp)

        # self._apply_penalty_vecrotized(x, self._weights)

    def _apply_penalty(self, j, wj_temp):

        if wj_temp > 0:
            self._weights[j] = max(0, wj_temp - (self.max_cum_l1 + self.cum_l1[j]))
        elif wj_temp < 0:
            self._weights[j] = min(0, wj_temp + (self.max_cum_l1 - self.cum_l1[j]))
        else:
            self._weights[j] = wj_temp

        # print(f"{self.cum_l1=}")
        self.cum_l1[j] = self.cum_l1[j] + (self._weights[j] - wj_temp)

        pass

    def _apply_penalty_vecrotized(self, w_temp):
        # REFLECTION: no way this is going to work;
        # can't return from numpy to VectorDict without iterating
        # REFLECTION: make it as is, as a stub, add a test and be done
        # TEST: this func weights on [-1] sample after normal training vs vanilla func weights on [:] training
        weights_mirror = self._weights.to_numpy(self._weights.keys())
        w_temp_mirror = w_temp.to_numpy(w_temp.keys())
        cum_l1_mirror = self.cum_l1.to_numpy(self.cum_l1.keys())
        # x_mirror = utils.VectorDict(x).to_numpy(utils.VectorDict(x).keys())

        indexer = w_temp_mirror > 0
        weights_mirror[indexer] = np.maximum(
            0, w_temp_mirror[indexer] - (self.max_cum_l1 + cum_l1_mirror[indexer])
        )

        indexer = w_temp_mirror < 0
        weights_mirror[indexer] = np.minimum(
            0, w_temp_mirror[indexer] + (self.max_cum_l1 - cum_l1_mirror[indexer])
        )

        cum_l1_mirror = cum_l1_mirror + (weights_mirror - w_temp_mirror)

        return weights_mirror, cum_l1_mirror

    # Single instance methods

    def _raw_dot_one(self, x: dict) -> float:
        return self._weights @ utils.VectorDict(x) + self.intercept

    def _eval_gradient_one(self, x: dict, y: float, w: float) -> (dict, float):

        loss_gradient = self.loss.gradient(y_true=y, y_pred=self._raw_dot_one(x))
        loss_gradient *= w
        loss_gradient = float(
            utils.math.clamp(loss_gradient, -self.clip_gradient, self.clip_gradient)
        )

        # print(loss_gradient)
        # print(
        #     loss_gradient * utils.VectorDict(x) + self.l2 * self._weights,
        # )
        # print("---")

        # self.max_cum_l1 = self.max_cum_l1 + self.l1_reg * self.learning_rate

        # for j, xj in x.items():
        #     wj_temp = self.weights[j] + self.learning_rate * xj * loss_gradient
        #     if wj_temp > 0:
        #         self.weights[j] = max(0, wj_temp - (self.max_cum_l1 + self.cum_l1[j]))
        #     elif wj_temp < 0:
        #         self.weights[j] = min(0, wj_temp + (self.max_cum_l1 - self.cum_l1[j]))
        #     else:
        #         self.weights[j] = wj_temp

        #     self.cum_l1[j] = self.cum_l1[j] + (self.weights[j] - wj_temp)
        if self.l1 == 0.0:
            return (
                loss_gradient * utils.VectorDict(x) + self.l2 * self._weights,
                loss_gradient,
            )
        if (self.l1 != 0.0) and (self.l2 == 0.0):
            # print("hello")
            return (
                loss_gradient * utils.VectorDict(x),
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

        loss_gradient = self.loss.gradient(y_true=y.values, y_pred=self._raw_dot_many(X))
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
