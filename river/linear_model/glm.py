import contextlib
import numbers
import typing

import numpy as np
import pandas as pd

from river import base
from river import optim
from river import utils


__all__ = ["LinearRegression", "LogisticRegression", "Perceptron"]


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

    def _fit(self, x, y, w, get_grad):

        # Some optimizers need to do something before a prediction is made
        self.optimizer.update_before_pred(w=self._weights)

        # Calculate the gradient
        gradient, loss_gradient = get_grad(x, y, w)

        # Update the intercept
        self.intercept -= self.intercept_lr.get(self.optimizer.n_iterations) * loss_gradient

        # Update the weights
        self.optimizer.update_after_pred(w=self._weights, g=gradient)

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
            loss_gradient * utils.VectorDict(x) + 2.0 * self.l2 * self._weights,
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

    def learn_many(self, X: pd.DataFrame, y: pd.Series, w: typing.Union[float, pd.Series] = 1):
        self._y_name = y.name
        with self._learn_mode(set(X)):
            return self._fit(X, y, w, get_grad=self._eval_gradient_many)


class LinearRegression(GLM, base.MiniBatchRegressor):
    """Linear regression.

    This estimator supports learning with mini-batches. On top of the single instance methods, it
    provides the following methods: `learn_many`, `predict_many`, `predict_proba_many`. Each method
    takes as input a `pandas.DataFrame` where each column represents a feature.

    It is generally a good idea to scale the data beforehand in order for the optimizer to
    converge. You can do this online with a `preprocessing.StandardScaler`.

    Parameters
    ----------
    optimizer
        The sequential optimizer used for updating the weights. Note that the intercept updates are
        handled separately.
    loss
        The loss function to optimize for.
    l2
        Amount of L2 regularization used to push weights towards 0.
    intercept_init
        Initial intercept value.
    intercept_lr
        Learning rate scheduler used for updating the intercept. A `optim.schedulers.Constant` is
        used if a `float` is provided. The intercept is not updated when this is set to 0.
    clip_gradient
        Clips the absolute value of each gradient value.
    initializer
        Weights initialization scheme.

    Attributes
    ----------
    weights : dict
        The current weights.

    Examples
    --------

    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import linear_model
    >>> from river import metrics
    >>> from river import preprocessing

    >>> dataset = datasets.TrumpApproval()

    >>> model = (
    ...     preprocessing.StandardScaler() |
    ...     linear_model.LinearRegression(intercept_lr=.1)
    ... )
    >>> metric = metrics.MAE()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    MAE: 0.555971

    >>> model['LinearRegression'].intercept
    35.617670

    You can call the `debug_one` method to break down a prediction. This works even if the
    linear regression is part of a pipeline.

    >>> x, y = next(iter(dataset))
    >>> report = model.debug_one(x)
    >>> print(report)
    0. Input
    --------
    gallup: 43.84321 (float)
    ipsos: 46.19925 (float)
    morning_consult: 48.31875 (float)
    ordinal_date: 736389 (int)
    rasmussen: 44.10469 (float)
    you_gov: 43.63691 (float)
    <BLANKLINE>
    1. StandardScaler
    -----------------
    gallup: 1.18810 (float)
    ipsos: 2.10348 (float)
    morning_consult: 2.73545 (float)
    ordinal_date: -1.73032 (float)
    rasmussen: 1.26872 (float)
    you_gov: 1.48391 (float)
    <BLANKLINE>
    2. LinearRegression
    -------------------
    Name              Value      Weight      Contribution
          Intercept    1.00000    35.61767       35.61767
              ipsos    2.10348     0.62689        1.31866
    morning_consult    2.73545     0.24180        0.66144
             gallup    1.18810     0.43568        0.51764
          rasmussen    1.26872     0.28118        0.35674
            you_gov    1.48391     0.03123        0.04634
       ordinal_date   -1.73032     3.45162       -5.97242
    <BLANKLINE>
    Prediction: 32.54607

    """

    def __init__(
        self,
        optimizer: optim.Optimizer = None,
        loss: optim.losses.RegressionLoss = None,
        l2=0.0,
        intercept_init=0.0,
        intercept_lr: typing.Union[optim.schedulers.Scheduler, float] = 0.01,
        clip_gradient=1e12,
        initializer: optim.initializers.Initializer = None,
    ):
        super().__init__(
            optimizer=optim.SGD(0.01) if optimizer is None else optimizer,
            loss=optim.losses.Squared() if loss is None else loss,
            intercept_init=intercept_init,
            intercept_lr=intercept_lr,
            l2=l2,
            clip_gradient=clip_gradient,
            initializer=initializer if initializer else optim.initializers.Zeros(),
        )

    def predict_one(self, x):
        return self.loss.mean_func(self._raw_dot_one(x))

    def predict_many(self, X):
        return pd.Series(
            self.loss.mean_func(self._raw_dot_many(X)),
            index=X.index,
            name=self._y_name,
            copy=False,
        )

    def debug_one(self, x: dict, decimals=5) -> str:
        """Debugs the output of the linear regression.

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

        def fmt_float(x):
            return "{: ,.{prec}f}".format(x, prec=decimals)

        names = list(map(str, x.keys())) + ["Intercept"]
        values = list(map(fmt_float, list(x.values()) + [1]))
        weights = list(map(fmt_float, [self._weights.get(i, 0) for i in x] + [self.intercept]))
        contributions = [xi * self._weights.get(i, 0) for i, xi in x.items()] + [self.intercept]
        order = reversed(np.argsort(contributions))
        contributions = list(map(fmt_float, contributions))

        table = utils.pretty.print_table(
            headers=["Name", "Value", "Weight", "Contribution"],
            columns=[names, values, weights, contributions],
            order=order,
        )

        return table


class LogisticRegression(GLM, base.MiniBatchClassifier):
    """Logistic regression.

    This estimator supports learning with mini-batches. On top of the single instance methods, it
    provides the following methods: `learn_many`, `predict_many`, `predict_proba_many`. Each method
    takes as input a `pandas.DataFrame` where each column represents a feature.

    It is generally a good idea to scale the data beforehand in order for the optimizer to
    converge. You can do this online with a `preprocessing.StandardScaler`.

    Parameters
    ----------
    optimizer
        The sequential optimizer used for updating the weights. Note that the intercept is handled
        separately.
    loss
        The loss function to optimize for. Defaults to `optim.losses.Log`.
    l2
        Amount of L2 regularization used to push weights towards 0.
    intercept_init
        Initial intercept value.
    intercept_lr
        Learning rate scheduler used for updating the intercept. A `optim.schedulers.Constant` is
        used if a `float` is provided. The intercept is not updated when this is set to 0.
    clip_gradient
        Clips the absolute value of each gradient value.
    initializer
        Weights initialization scheme.

    Attributes
    ----------
    weights
        The current weights.

    Examples
    --------

    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import linear_model
    >>> from river import metrics
    >>> from river import optim
    >>> from river import preprocessing

    >>> dataset = datasets.Phishing()

    >>> model = (
    ...     preprocessing.StandardScaler() |
    ...     linear_model.LogisticRegression(optimizer=optim.SGD(.1))
    ... )

    >>> metric = metrics.Accuracy()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    Accuracy: 88.96%

    """

    def __init__(
        self,
        optimizer: optim.Optimizer = None,
        loss: optim.losses.BinaryLoss = None,
        l2=0.0,
        intercept_init=0.0,
        intercept_lr: typing.Union[float, optim.schedulers.Scheduler] = 0.01,
        clip_gradient=1e12,
        initializer: optim.initializers.Initializer = None,
    ):

        super().__init__(
            optimizer=optim.SGD(0.01) if optimizer is None else optimizer,
            loss=optim.losses.Log() if loss is None else loss,
            intercept_init=intercept_init,
            intercept_lr=intercept_lr,
            l2=l2,
            clip_gradient=clip_gradient,
            initializer=initializer if initializer else optim.initializers.Zeros(),
        )

    def predict_proba_one(self, x):
        p = self.loss.mean_func(self._raw_dot_one(x))  # Convert logit to probability
        return {False: 1.0 - p, True: p}

    def predict_proba_many(self, X: pd.DataFrame) -> pd.DataFrame:
        p = self.loss.mean_func(self._raw_dot_many(X))  # Convert logits to probabilities
        return pd.DataFrame({False: 1.0 - p, True: p}, index=X.index, copy=False)


class Perceptron(LogisticRegression):
    """Perceptron classifier.

    In this implementation, the Perceptron is viewed as a special case of the logistic regression.
    The loss function that is used is the Hinge loss with a threshold set to 0, whilst the learning
    rate of the stochastic gradient descent procedure is set to 1 for both the weights and the
    intercept.

    Parameters
    ----------
    l2
        Amount of L2 regularization used to push weights towards 0.
    clip_gradient
        Clips the absolute value of each gradient value.
    initializer
        Weights initialization scheme.

    Attributes
    ----------
    weights
        The current weights.

    Examples
    --------

    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import linear_model as lm
    >>> from river import metrics
    >>> from river import preprocessing as pp

    >>> dataset = datasets.Phishing()

    >>> model = pp.StandardScaler() | lm.Perceptron()

    >>> metric = metrics.Accuracy()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    Accuracy: 85.84%

    """

    def __init__(
        self,
        l2=0.0,
        clip_gradient=1e12,
        initializer: optim.initializers.Initializer = None,
    ):
        super().__init__(
            optimizer=optim.SGD(1),
            intercept_lr=1,
            loss=optim.losses.Hinge(threshold=0.0),
            l2=l2,
            clip_gradient=clip_gradient,
            initializer=initializer,
        )
