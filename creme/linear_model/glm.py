import collections
import math
import numbers
import typing

import numpy as np
import pandas as pd

from creme import base
from creme import optim
from creme import utils


__all__ = [
    'LinearRegression',
    'LogisticRegression'
]


class GLM:
    """Generalized Linear Model.

    This serves as a base class for linear and logistic regression.

    """

    def __init__(self, optimizer, loss, l2, intercept, intercept_lr, clip_gradient, initializer):
        self.optimizer = optimizer
        self.loss = loss
        self.l2 = l2
        self.intercept = intercept
        self.intercept_lr = (
            optim.schedulers.Constant(intercept_lr)
            if isinstance(intercept_lr, numbers.Number) else
            intercept_lr
        )
        self.clip_gradient = clip_gradient
        self.weights = collections.defaultdict(initializer)
        self.initializer = initializer

        # The predict_many functions are going to return pandas.Series. We can name the series with
        # the name given to the y series seen during the last fit_many call.
        self._y_name = None

    def _fit(self, x, y, w, get_grad):

        # Some optimizers need to do something before a prediction is made
        self.weights = self.optimizer.update_before_pred(w=self.weights)

        # Calculate the gradient
        gradient, loss_gradient = get_grad(x, y, w)

        # Update the intercept
        self.intercept -= self.intercept_lr.get(self.optimizer.n_iterations) * loss_gradient

        # Update the weights
        self.weights = self.optimizer.update_after_pred(w=self.weights, g=gradient)

        return self

    # Single instance methods

    def _raw_dot_one(self, x: dict) -> float:
        return utils.math.dot(self.weights, x) + self.intercept

    def _eval_gradient_one(self, x: dict, y: float, w: float) -> (dict, float):

        loss_gradient = self.loss.gradient(y_true=y, y_pred=self._raw_dot_one(x))
        loss_gradient *= w
        loss_gradient = np.clip(loss_gradient, -self.clip_gradient, self.clip_gradient)

        return (
            {
                i: xi * loss_gradient + 2. * self.l2 * self.weights.get(i, 0)
                for i, xi in x.items()
            },
            loss_gradient
        )

    def fit_one(self, x, y, w=1.):
        return self._fit(x, y, w, get_grad=self._eval_gradient_one)

    # Mini-batch methods

    def _raw_dot_many(self, X: pd.DataFrame) -> np.ndarray:
        weights = np.array([self.weights[c] for c in X.columns])
        return X.values @ weights + self.intercept

    def _eval_gradient_many(self,
                            X: pd.DataFrame,
                            y: pd.Series,
                            w: typing.Union[float, pd.Series]) -> (dict, float):

        loss_gradient = self.loss.gradient(y_true=y.values, y_pred=self._raw_dot_many(X))
        loss_gradient *= w
        loss_gradient = np.clip(loss_gradient, -self.clip_gradient, self.clip_gradient)

        # At this point we have a feature matrix X of shape (n, p). The loss gradient is a vector
        # of length p. We want to multiply each of X's by the corresponding value in the loss
        # gradient. When this is all done, we collapse X by computing the average of each column,
        # thereby obtaining the mean gradient of the batch. From thereon, the code reduces to the
        # single instance case.
        gradient = np.einsum('ij,i->ij', X.values, loss_gradient).mean(axis=0)

        return dict(zip(X.columns, gradient)), loss_gradient.mean()

    def fit_many(self, X: pd.DataFrame, y: pd.Series, w: typing.Union[float, pd.Series] = 1):
        self._y_name = y.name
        return self._fit(X, y, w, get_grad=self._eval_gradient_many)


class LinearRegression(GLM, base.Regressor):
    """Linear regression.

    This estimator supports learning with mini-batches. On top of the single instance methods, it
    provides the following methods: `fit_many`, `predict_many`, `predict_proba_many`. Each method
    takes as input a `pandas.DataFrame` where each column represents a feature.

    Parameters:
        optimizer: The sequential optimizer used for updating the weights. Note
            that the intercept is handled separately. Defaults to `optim.SGD(.01)`.
        loss: The loss function to optimize for. Defaults to `optim.losses.SquaredLoss`.
        l2: Amount of L2 regularization used to push weights towards 0.
        intercept: Initial intercept value.
        intercept_lr (optim.schedulers.Scheduler or float): Learning rate scheduler used for
            updating the intercept. If a `float` is passed, then an instance of
            `optim.schedulers.Constant` will be used. Setting this to 0 implies that the intercept
            will be not be updated.
        l2: Amount of L2 regularization used to push weights towards 0.
        clip_gradient: Clips the absolute value of each gradient value.
        initializer: Weights initialization scheme.

    Attributes:
        weights (collections.defaultdict): The current weights.

    Example:

        >>> from creme import datasets
        >>> from creme import linear_model
        >>> from creme import metrics
        >>> from creme import model_selection
        >>> from creme import preprocessing

        >>> X_y = datasets.TrumpApproval()

        >>> model = (
        ...     preprocessing.StandardScaler() |
        ...     linear_model.LinearRegression(intercept_lr=.1)
        ... )
        >>> metric = metrics.MAE()

        >>> model_selection.progressive_val_score(X_y, model, metric)
        MAE: 0.555971

        >>> model['LinearRegression'].intercept
        35.617670

        You can call the `debug_one` method to break down a prediction. This works even if the
        linear regression is part of a pipeline.

        >>> x, y = next(iter(X_y))
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

    .. tip::
        It is generally a good idea to use a `preprocessing.StandardScaler` to help the optimizer
        converge by scaling the input features.

    """

    def __init__(self, optimizer: optim.Optimizer = None, loss: optim.losses.RegressionLoss = None,
                 l2=.0, intercept=0.,
                 intercept_lr: typing.Union[optim.schedulers.Scheduler, float] = .01,
                 clip_gradient=1e+12, initializer: optim.initializers.Initializer = None):
        super().__init__(
            optimizer=optim.SGD(.01) if optimizer is None else optimizer,
            loss=optim.losses.Squared() if loss is None else loss,
            intercept=intercept,
            intercept_lr=intercept_lr,
            l2=l2,
            clip_gradient=clip_gradient,
            initializer=initializer if initializer else optim.initializers.Zeros()
        )

    def predict_one(self, x):
        return self.loss.mean_func(self._raw_dot_one(x))

    def predict_many(self, X):
        return pd.Series(
            self.loss.mean_func(self._raw_dot_many(X)),
            index=X.index,
            name=self._y_name,
            copy=False
        )

    def debug_one(self, x: dict, decimals=5) -> str:
        """Debugs the output of the linear regression.

        Parameters:
            x: A dictionary of features.
            decimals: The number of decimals use for printing each numeric value.

        Returns:
            A table which explains the output.

        """

        def fmt_float(x):
            return '{: ,.{prec}f}'.format(x, prec=decimals)

        names = list(map(str, x.keys())) + ['Intercept']
        values = list(map(fmt_float, list(x.values()) + [1]))
        weights = list(map(fmt_float, [self.weights.get(i, 0) for i in x] + [self.intercept]))
        contributions = [xi * self.weights.get(i, 0) for i, xi in x.items()] + [self.intercept]
        order = reversed(np.argsort(contributions))
        contributions = list(map(fmt_float, contributions))

        table = utils.pretty.print_table(
            headers=['Name', 'Value', 'Weight', 'Contribution'],
            columns=[names, values, weights, contributions],
            order=order
        )

        return table


class LogisticRegression(GLM, base.BinaryClassifier):
    """Logistic regression.

    This estimator supports learning with mini-batches. On top of the single instance methods, it
    provides the following methods: `fit_many`, `predict_many`, `predict_proba_many`. Each method
    takes as input a `pandas.DataFrame` where each column represents a feature.

    Parameters:
        optimizer: The sequential optimizer used for updating the weights. Note that the intercept
            is handled separately. Defaults to `optim.SGD(.05)`.
        loss: The loss function to optimize for. Defaults to `optim.losses.Log`.
        l2: Amount of L2 regularization used to push weights towards 0.
        intercept: Initial intercept value.
        intercept_lr: Learning rate scheduler used for updating the intercept. If a `float` is
            passed, then an instance of `optim.schedulers.Constant` will be used. Setting this to 0
            implies that the intercept will be not be updated.
        l2: Amount of L2 regularization used to push weights towards 0.
        clip_gradient: Clips the absolute value of each gradient value.
        initializer: Weights initialization scheme.

    Attributes:
        weights (collections.defaultdict): The current weights.

    Example:

        >>> from creme import datasets
        >>> from creme import linear_model
        >>> from creme import metrics
        >>> from creme import model_selection
        >>> from creme import optim
        >>> from creme import preprocessing

        >>> X_y = datasets.Phishing()

        >>> model = (
        ...     preprocessing.StandardScaler() |
        ...     linear_model.LogisticRegression(optimizer=optim.SGD(.1))
        ... )

        >>> metric = metrics.Accuracy()

        >>> model_selection.progressive_val_score(X_y, model, metric)
        Accuracy: 88.96%

    .. tip::
        It is generally a good idea to use a `preprocessing.StandardScaler` to help the optimizer
        converge by scaling the input features.

    """

    def __init__(self, optimizer: optim.Optimizer = None, loss: optim.losses.BinaryLoss = None,
                 l2=.0, intercept=0.,
                 intercept_lr: typing.Union[float, optim.schedulers.Scheduler] = .01,
                 clip_gradient=1e12, initializer: optim.initializers.Initializer = None):

        super().__init__(
            optimizer=optim.SGD(.01) if optimizer is None else optimizer,
            loss=optim.losses.Log() if loss is None else loss,
            intercept=intercept,
            intercept_lr=intercept_lr,
            l2=l2,
            clip_gradient=clip_gradient,
            initializer=initializer if initializer else optim.initializers.Zeros()
        )

    def predict_proba_one(self, x):
        p = self.loss.mean_func(self._raw_dot_one(x))  # Convert logit to probability
        return {False: 1. - p, True: p}

    def predict_proba_many(self, X: pd.DataFrame) -> pd.DataFrame:
        p = self.loss.mean_func(self._raw_dot_many(X))  # Convert logits to probabilities
        return pd.DataFrame({False: 1. - p, True: p}, index=X.index, copy=False)

    def predict_many(self, X: pd.DataFrame) -> pd.Series:
        p = self.loss.mean_func(self._raw_dot_many(X))
        return pd.Series(p > .5, name=self._y_name, index=X.index, copy=False)
