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
        intercept (float): Initial intercept value.
        intercept_lr (optim.schedulers.Scheduler or float): Learning rate scheduler used for
            updating the intercept. If a `float` is passed, then an instance of
            `optim.schedulers.Constant` will be used. Setting this to 0 implies that the intercept
            will be not be updated. Setting this to 0 means that no intercept will be used.
        l2 (float): Amount of L2 regularization used to push weights towards 0.
        clip_gradient (float): Clips the absolute value of each gradient value.

    Attributes:
        weights (collections.defaultdict)

    """

    def __init__(self, optimizer, loss, l2, intercept, intercept_lr, clip_gradient):
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
        self.weights = collections.defaultdict(float)

    def _raw_dot(self, x):
        return utils.math.dot(self.weights, x) + self.intercept

    def fit_one(self, x, y, sample_weight=1.):

        # Some optimizers need to do something before a prediction is made
        self.weights = self.optimizer.update_before_pred(w=self.weights)

        # Obtain the gradient of the loss with respect to the raw output
        g_loss = self.loss.gradient(y_true=y, y_pred=self._raw_dot(x))

        # Clamp the gradient to avoid numerical instability
        g_loss = utils.math.clamp(g_loss, minimum=-self.clip_gradient, maximum=self.clip_gradient)

        # Apply the sample weight
        g_loss *= sample_weight

        # Calculate the gradient
        gradient = {
            i: xi * g_loss + 2. * self.l2 * self.weights.get(i, 0)
            for i, xi in x.items()
        }

        # Update the intercept
        self.intercept -= self.intercept_lr.get(self.optimizer.n_iterations) * g_loss

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
        intercept (float): Initial intercept value.
        intercept_lr (optim.schedulers.Scheduler or float): Learning rate scheduler used for
            updating the intercept. If a `float` is passed, then an instance of
            `optim.schedulers.Constant` will be used. Setting this to 0 implies that the intercept
            will be not be updated.
        l2 (float): Amount of L2 regularization used to push weights towards 0.

    Attributes:
        weights (collections.defaultdict): The current weights assigned to the features.

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
            >>> metric = metrics.MAE()

            >>> model_selection.online_score(X_y, model, metric)
            MAE: 4.038404

            >>> model['LinearRegression'].intercept
            22.189736...

    Note:
        Using a feature scaler such as `preprocessing.StandardScaler` upstream helps the optimizer
        to converge.

    """

    def __init__(self, optimizer=None, loss=None, l2=.0001, intercept=0., intercept_lr=.01,
                 clip_gradient=1e12):
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
            clip_gradient=clip_gradient
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
                Intercept    1.00000    22.18974       22.18974
                      DIS   -0.51305    -1.82199        0.93478
                    LSTAT   -0.28330    -3.01991        0.85554
                       RM    0.17131     3.45826        0.59244
                     CRIM   -0.39351    -0.68585        0.26989
                      NOX   -0.29941    -0.57048        0.17081
                      TAX   -0.14381    -0.30284        0.04355
                    INDUS   -0.37560     0.08929       -0.03354
                      AGE    0.59772    -0.08945       -0.05346
                       ZN   -0.48724     0.47388       -0.23089
                     CHAS   -0.27233     1.14375       -0.31148
                      RAD   -0.52248     0.70101       -0.36627
                  PTRATIO    1.12911    -1.61350       -1.82182
                        B   -3.13133     1.13608       -3.55744
                <BLANKLINE>
                Prediction: 18.68184

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
        intercept (float): Initial intercept value.
        intercept_lr (optim.schedulers.Scheduler or float): Learning rate scheduler used for
            updating the intercept. If a `float` is passed, then an instance of
            `optim.schedulers.Constant` will be used. Setting this to 0 implies that the intercept
            will be not be updated.
        l2 (float): Amount of L2 regularization used to push weights towards .

    Attributes:
        weights (collections.defaultdict)

    Example:

        ::

            >>> from creme import datasets
            >>> from creme import linear_model
            >>> from creme import metrics
            >>> from creme import model_selection
            >>> from creme import optim
            >>> from creme import preprocessing

            >>> X_y = datasets.fetch_electricity()

            >>> model = (
            ...     preprocessing.StandardScaler() |
            ...     linear_model.LogisticRegression(optimizer=optim.SGD(.1))
            ... )
            >>> metric = metrics.Accuracy()

            >>> model_selection.online_score(X_y, model, metric)
            Accuracy: 89.46%

    Note:
        Using a feature scaler such as `preprocessing.StandardScaler` upstream helps the optimizer
        to converge.

    """

    def __init__(self, optimizer=None, loss=None, l2=.0001, intercept=0., intercept_lr=.01,
                 clip_gradient=1e12):
        super().__init__(
            optimizer=optim.SGD(.01) if optimizer is None else optimizer,
            loss=optim.losses.Log() if loss is None else loss,
            intercept=intercept,
            intercept_lr=intercept_lr,
            l2=l2,
            clip_gradient=clip_gradient
        )

    def predict_proba_one(self, x):
        p = utils.math.sigmoid(self._raw_dot(x))  # Convert logit to probability
        return {False: 1. - p, True: p}
