import collections
import math
import numbers

import numpy as np

from .. import base
from .. import optim
from .. import utils


__all__ = [
    'LinearRegression',
    'LogisticRegression'
]


class GLM:
    """Generalized Linear Model.

    Parameters:
        optimizer (optim.Optimizer): The sequential optimizer used for updating the weights. Note
            that the intercept is handled separately.
        loss (optim.Loss): The loss function to optimize for.
        l2 (float): Amount of L2 regularization used to push weights towards 0.
        intercept (float): Initial intercept value.
        intercept_lr (optim.schedulers.Scheduler or float): Learning rate scheduler used for
            updating the intercept. If a `float` is passed, then an instance of
            `optim.schedulers.Constant` will be used. Setting this to 0 implies that the intercept
            will be not be updated.
        clip_gradient (float): Clips the absolute value of each gradient value.
        initializer (optim.initializers.Initializer): Weights initialization scheme.

    Attributes:
        weights (collections.defaultdict): The current weights.

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

    def _raw_dot(self, x):
        return utils.math.dot(self.weights, x) + self.intercept

    def _eval_gradient(self, x, y, sample_weight):
        """Returns the gradient for a given observation.

        This logic is put into a separate function for testing purposes.

        """

        loss_gradient = self.loss.gradient(y_true=y, y_pred=self._raw_dot(x))

        # Apply the sample weight
        loss_gradient *= sample_weight

        # Clip the gradient to avoid numerical instability
        loss_gradient = utils.math.clamp(
            loss_gradient,
            minimum=-self.clip_gradient,
            maximum=self.clip_gradient
        )

        return (
            {
                i: (
                    xi * loss_gradient +
                    2. * self.l2 * self.weights.get(i, 0)
                )
                for i, xi in x.items()
            },
            loss_gradient
        )

    def fit_one(self, x, y, sample_weight=1.):

        # Some optimizers need to do something before a prediction is made
        self.weights = self.optimizer.update_before_pred(w=self.weights)

        # Calculate the gradient
        gradient, loss_gradient = self._eval_gradient(x=x, y=y, sample_weight=sample_weight)

        # Update the intercept
        self.intercept -= self.intercept_lr.get(self.optimizer.n_iterations) * loss_gradient

        # Update the weights
        self.weights = self.optimizer.update_after_pred(w=self.weights, g=gradient)

        return self


class LinearRegression(GLM, base.Regressor):
    """Linear regression.

    Parameters:
        optimizer (optim.Optimizer): The sequential optimizer used for updating the weights. Note
            that the intercept is handled separately. Defaults to ``optim.SGD(.01)``.
        loss (optim.RegressionLoss): The loss function to optimize for. Defaults to
            ``optim.losses.SquaredLoss``.
        l2 (float): Amount of L2 regularization used to push weights towards 0.
        intercept (float): Initial intercept value.
        intercept_lr (optim.schedulers.Scheduler or float): Learning rate scheduler used for
            updating the intercept. If a `float` is passed, then an instance of
            `optim.schedulers.Constant` will be used. Setting this to 0 implies that the intercept
            will be not be updated.
        l2 (float): Amount of L2 regularization used to push weights towards 0.
        clip_gradient (float): Clips the absolute value of each gradient value.
        initializer (optim.initializers.Initializer): Weights initialization scheme.

    Attributes:
        weights (collections.defaultdict): The current weights.

    Example:

        ::

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
            MAE: 0.616405

            >>> model['LinearRegression'].intercept
            37.966291...

    Note:
        Using a feature scaler such as `preprocessing.StandardScaler` upstream helps the optimizer
        to converge.

    """

    def __init__(self, optimizer=None, loss=None, l2=.0, intercept=0., intercept_lr=.01,
                 clip_gradient=1e12, initializer=None):
        super().__init__(
            optimizer=(
                optim.SGD(optim.schedulers.InverseScaling(.01, .25))
                if optimizer is None else
                optimizer
            ),
            loss=optim.losses.Squared() if loss is None else loss,
            intercept=intercept,
            intercept_lr=intercept_lr,
            l2=l2,
            clip_gradient=clip_gradient,
            initializer=initializer if initializer else optim.initializers.Zeros()
        )

    def predict_one(self, x):
        return self.loss.mean_func(self._raw_dot(x))

    def debug_one(self, x, decimals=5, **print_params):
        """

        Example:

            ::

                >>> from creme import linear_model
                >>> from creme import metrics
                >>> from creme import model_selection
                >>> from creme import preprocessing
                >>> from creme import stream
                >>> from sklearn import datasets

                >>> X_y = stream.iter_sklearn_dataset(
                ...     dataset=datasets.load_boston(),
                ...     shuffle=True,
                ...     seed=42
                ... )

                >>> model = (
                ...     preprocessing.StandardScaler() |
                ...     linear_model.LinearRegression(intercept_lr=.1)
                ... )

                >>> for x, y in X_y:
                ...     y_pred = model.predict_one(x)
                ...     model = model.fit_one(x, y)

                >>> model.debug_one(x)
                ... # doctest: +NORMALIZE_WHITESPACE
                0. Input
                --------
                AGE: 83.40000 (float64)
                B: 395.43000 (float64)
                CHAS: 1.00000 (float64)
                CRIM: 5.20177 (float64)
                DIS: 2.72270 (float64)
                INDUS: 18.10000 (float64)
                LSTAT: 11.48000 (float64)
                NOX: 0.77000 (float64)
                PTRATIO: 20.20000 (float64)
                RAD: 24.00000 (float64)
                RM: 6.12700 (float64)
                TAX: 666.00000 (float64)
                ZN: 0.00000 (float64)
                <BLANKLINE>
                1. StandardScaler
                -----------------
                AGE: 0.52667 (float64)
                B: 0.42451 (float64)
                CHAS: 3.66477 (float64)
                CRIM: 0.18465 (float64)
                DIS: -0.50925 (float64)
                INDUS: 1.01499 (float64)
                LSTAT: -0.16427 (float64)
                NOX: 1.85804 (float64)
                PTRATIO: 0.80578 (float64)
                RAD: 1.65960 (float64)
                RM: -0.22435 (float64)
                TAX: 1.52941 (float64)
                ZN: -0.48724 (float64)
                <BLANKLINE>
                2. LinearRegression
                -------------------
                Name        Value      Weight      Contribution
                Intercept    1.00000    21.30237       21.30237
                    CHAS    3.66477     0.80408        2.94679
                    RAD    1.65960     0.58526        0.97129
                    DIS   -0.50925    -1.57885        0.80404
                    LSTAT   -0.16427    -3.16813        0.52043
                        B    0.42451     1.05251        0.44681
                    CRIM    0.18465    -0.88817       -0.16400
                    ZN   -0.48724     0.43327       -0.21111
                    AGE    0.52667    -0.45398       -0.23909
                    INDUS    1.01499    -0.35933       -0.36471
                    RM   -0.22435     3.41237       -0.76558
                    TAX    1.52941    -0.64650       -0.98876
                    NOX    1.85804    -0.59396       -1.10361
                PTRATIO    0.80578    -1.78695       -1.43988
                <BLANKLINE>
                Prediction: 21.71498

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

        print(table, **print_params)


class LogisticRegression(GLM, base.BinaryClassifier):
    """Logistic regression.

    Parameters:
        optimizer (optim.Optimizer): The sequential optimizer used for updating the weights. Note
            that the intercept is handled separately. Defaults to ``optim.SGD(.05)``.
        loss (optim.BinaryLoss): The loss function to optimize for. Defaults to
            ``optim.losses.Log``.
        l2 (float): Amount of L2 regularization used to push weights towards 0.
        intercept (float): Initial intercept value.
        intercept_lr (optim.schedulers.Scheduler or float): Learning rate scheduler used for
            updating the intercept. If a `float` is passed, then an instance of
            `optim.schedulers.Constant` will be used. Setting this to 0 implies that the intercept
            will be not be updated.
        l2 (float): Amount of L2 regularization used to push weights towards 0.
        clip_gradient (float): Clips the absolute value of each gradient value.
        initializer (optim.initializers.Initializer): Weights initialization scheme.

    Attributes:
        weights (collections.defaultdict): The current weights.

    Example:

        ::

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

    Note:
        Using a feature scaler such as `preprocessing.StandardScaler` upstream helps the optimizer
        to converge.

    """

    def __init__(self, optimizer=None, loss=None, l2=.0, intercept=0., intercept_lr=.01,
                 clip_gradient=1e12, initializer=None):
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
        p = self.loss.mean_func(self._raw_dot(x))  # Convert logit to probability
        return {False: 1. - p, True: p}
