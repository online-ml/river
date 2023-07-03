from __future__ import annotations

import numpy as np
import pandas as pd

from river import base, linear_model, optim, utils


class LinearRegression(linear_model.base.GLM, base.MiniBatchRegressor):
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
    MAE: 0.558735

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
        optimizer: optim.base.Optimizer | None = None,
        loss: optim.losses.RegressionLoss | None = None,
        l2=0.0,
        l1=0.0,
        intercept_init=0.0,
        intercept_lr: optim.base.Scheduler | float = 0.01,
        clip_gradient=1e12,
        initializer: optim.base.Initializer | None = None,
    ):
        super().__init__(
            optimizer=optim.SGD(0.01) if optimizer is None else optimizer,
            loss=optim.losses.Squared() if loss is None else loss,
            intercept_init=intercept_init,
            intercept_lr=intercept_lr,
            l2=l2,
            l1=l1,
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

    def debug_one(self, x: dict, decimals: int = 5) -> str:
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
        order = list(reversed(np.argsort(contributions)))
        contributions = list(map(fmt_float, contributions))

        table = utils.pretty.print_table(
            headers=["Name", "Value", "Weight", "Contribution"],
            columns=[names, values, weights, contributions],
            order=order,
        )

        return table
