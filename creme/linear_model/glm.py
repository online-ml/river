import collections
import numbers

from .. import base
from .. import optim
from .. import utils


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
            will be not be updated. Setting this to 0 means that no intercept will be used.
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

    def _eval_gradient(self, x, y, sample_weight=1.):
        """Returns the gradient for a given observation.

        This logic is put into a separate function for testing purposes.

        """

        loss_gradient = self.loss.gradient(y_true=y, y_pred=self._raw_dot(x))

        # Clip the gradient to avoid numerical instability
        loss_gradient = utils.math.clamp(
            loss_gradient,
            minimum=-self.clip_gradient,
            maximum=self.clip_gradient
        )

        # Apply the sample weight
        loss_gradient *= sample_weight

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
            ``optim.SquaredLoss``.
        l2 (float): Amount of L2 regularization used to push weights towards 0.
        intercept (float): Initial intercept value.
        intercept_lr (optim.schedulers.Scheduler or float): Learning rate scheduler used for
            updating the intercept. If a `float` is passed, then an instance of
            `optim.schedulers.Constant` will be used. Setting this to 0 implies that the intercept
            will be not be updated. Setting this to 0 means that no intercept will be used.
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
        return self._raw_dot(x)

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
                ...     random_state=42
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
                AGE: 85.40000 (float64)
                B: 70.80000 (float64)
                CHAS: 0.00000 (float64)
                CRIM: 0.22876 (float64)
                DIS: 2.71470 (float64)
                INDUS: 8.56000 (float64)
                LSTAT: 10.63000 (float64)
                NOX: 0.52000 (float64)
                PTRATIO: 20.90000 (float64)
                RAD: 5.00000 (float64)
                RM: 6.40500 (float64)
                TAX: 384.00000 (float64)
                ZN: 0.00000 (float64)
                <BLANKLINE>
                1. StandardScaler
                -----------------
                AGE: 0.59772 (float64)
                B: -3.13133 (float64)
                CHAS: -0.27233 (float64)
                CRIM: -0.39351 (float64)
                DIS: -0.51305 (float64)
                INDUS: -0.37560 (float64)
                LSTAT: -0.28330 (float64)
                NOX: -0.29941 (float64)
                PTRATIO: 1.12911 (float64)
                RAD: -0.52248 (float64)
                RM: 0.17131 (float64)
                TAX: -0.14381 (float64)
                ZN: -0.48724 (float64)
                <BLANKLINE>
                2. LinearRegression
                -------------------
                Name        Value      Weight      Contribution
                Intercept    1.00000    22.18990       22.18990
                      DIS   -0.51305    -1.82221        0.93489
                    LSTAT   -0.28330    -3.02012        0.85560
                       RM    0.17131     3.45849        0.59247
                     CRIM   -0.39351    -0.68592        0.26991
                      NOX   -0.29941    -0.57055        0.17083
                      TAX   -0.14381    -0.30286        0.04355
                    INDUS   -0.37560     0.08940       -0.03358
                      AGE    0.59772    -0.08943       -0.05346
                       ZN   -0.48724     0.47393       -0.23092
                     CHAS   -0.27233     1.14376       -0.31148
                      RAD   -0.52248     0.70114       -0.36633
                  PTRATIO    1.12911    -1.61355       -1.82188
                        B   -3.13133     1.13617       -3.55773
                <BLANKLINE>
                Prediction: 18.68179

        """

        def fmt_float(x):
            return '{: ,.{prec}f}'.format(x, prec=decimals)

        names = list(map(str, x.keys())) + ['Intercept']
        values = list(map(fmt_float, x.values())) + [fmt_float(1)]
        weights = list(map(fmt_float, self.weights.values())) + [fmt_float(self.intercept)]
        contributions = (
            [fmt_float(xi * self.weights[i]) for i, xi in x.items()] +
            [fmt_float(self.intercept)]
        )

        table = utils.pretty.print_table(
            headers=['Name', 'Value', 'Weight', 'Contribution'],
            columns=[names, values, weights, contributions],
            sort_by='Contribution'
        )

        print(table, **print_params)


class LogisticRegression(GLM, base.BinaryClassifier):
    """Logistic regression.

    Parameters:
        optimizer (optim.Optimizer): The sequential optimizer used for updating the weights. Note
            that the intercept is handled separately. Defaults to ``optim.SGD(.05)``.
        loss (optim.BinaryLoss): The loss function to optimize for. Defaults to ``optim.LogLoss``.
        l2 (float): Amount of L2 regularization used to push weights towards 0.
        intercept (float): Initial intercept value.
        intercept_lr (optim.schedulers.Scheduler or float): Learning rate scheduler used for
            updating the intercept. If a `float` is passed, then an instance of
            `optim.schedulers.Constant` will be used. Setting this to 0 implies that the intercept
            will be not be updated. Setting this to 0 means that no intercept will be used.
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

            >>> X_y = datasets.Elec2()

            >>> model = (
            ...     preprocessing.StandardScaler() |
            ...     linear_model.LogisticRegression(optimizer=optim.SGD(.1))
            ... )
            >>> metric = metrics.Accuracy()

            >>> model_selection.progressive_val_score(X_y, model, metric)
            Accuracy: 89.49%

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
        p = utils.math.sigmoid(self._raw_dot(x))  # Convert logit to probability
        return {False: 1. - p, True: p}
