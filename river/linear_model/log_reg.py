from __future__ import annotations

import pandas as pd

from river import base, linear_model, optim


class LogisticRegression(linear_model.base.GLM, base.MiniBatchClassifier):
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
        optimizer: optim.base.Optimizer | None = None,
        loss: optim.losses.BinaryLoss | None = None,
        l2=0.0,
        l1=0.0,
        intercept_init=0.0,
        intercept_lr: float | optim.base.Scheduler = 0.01,
        clip_gradient=1e12,
        initializer: optim.base.Initializer | None = None,
    ):
        super().__init__(
            optimizer=optim.SGD(0.01) if optimizer is None else optimizer,
            loss=optim.losses.Log() if loss is None else loss,
            intercept_init=intercept_init,
            intercept_lr=intercept_lr,
            l2=l2,
            l1=l1,
            clip_gradient=clip_gradient,
            initializer=initializer if initializer else optim.initializers.Zeros(),
        )

    def predict_proba_one(self, x):
        p = self.loss.mean_func(self._raw_dot_one(x))  # Convert logit to probability
        return {False: 1.0 - p, True: p}

    def predict_proba_many(self, X: pd.DataFrame) -> pd.DataFrame:
        p = self.loss.mean_func(self._raw_dot_many(X))  # Convert logits to probabilities
        return pd.DataFrame({False: 1.0 - p, True: p}, index=X.index, copy=False)
