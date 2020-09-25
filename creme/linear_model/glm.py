import collections
import itertools
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
    
    def _eval_hessian(self, x, y):
        """Returns the hessian for a given observation.

        This logic is put into a separate function for testing purposes.

        """
        
        # Hessian = x 
        
        # Hessian = utils.math.outer(x, x)
        
        Hessian = {}
        for (i, xi), (j, xj) in itertools.product(x.items(), x.items()):
            Hessian[i, j] = xi * xj * self.loss.hessian(y_true=y, y_pred=self._raw_dot(x))

        return Hessian
            
    def fit_one(self, x, y, sample_weight=1.):

        # Some optimizers need to do something before a prediction is made
        self.weights = self.optimizer.update_before_pred(w=self.weights)

        # Calculate the gradient
        gradient, loss_gradient = self._eval_gradient(x=x, y=y, sample_weight=sample_weight)

        # Update the intercept
        self.intercept -= self.intercept_lr.get(self.optimizer.n_iterations) * loss_gradient

        # Update the weights
        self.weights = self.optimizer.update_after_pred(w=self.weights, g=gradient, h=self._eval_hessian(x, y))

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
            38.000439

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

                >>> for x, y in X_y:
                ...     y_pred = model.predict_one(x)
                ...     model = model.fit_one(x, y)

                >>> model.debug_one(x)
                0. Input
                --------
                gallup: 43.84321 (float)
                ipsos: 40.57068 (float)
                morning_consult: 37.81875 (float)
                ordinal_date: 737389 (int)
                rasmussen: 40.10469 (float)
                you_gov: 41.63691 (float)
                <BLANKLINE>
                1. StandardScaler
                -----------------
                gallup: 1.18751 (float)
                ipsos: -0.04683 (float)
                morning_consult: -1.22583 (float)
                ordinal_date: 1.72946 (float)
                rasmussen: -0.23857 (float)
                you_gov: 0.44131 (float)
                <BLANKLINE>
                2. LinearRegression
                -------------------
                Name              Value      Weight      Contribution
                      Intercept    1.00000    38.00044       38.00044
                   ordinal_date    1.72946     2.23125        3.85885
                         gallup    1.18751     0.28647        0.34019
                        you_gov    0.44131    -0.01270       -0.00560
                          ipsos   -0.04683     1.01815       -0.04768
                      rasmussen   -0.23857     0.45099       -0.10759
                morning_consult   -1.22583     0.35181       -0.43126
                <BLANKLINE>
                Prediction: 41.60735

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
