from __future__ import annotations

import pandas as pd

from river import anomaly, linear_model, optim


class OneClassSVM(linear_model.base.GLM, anomaly.base.AnomalyDetector):
    """One-class SVM for anomaly detection.

    This is a stochastic implementation of the one-class SVM algorithm, and will not exactly match
    its batch formulation.

    It is encouraged to scale the data upstream with `preprocessing.StandardScaler`, as well as use
    `feature_extraction.RBFSampler` to capture non-linearities.

    Parameters
    ----------
    nu
        An upper bound on the fraction of training errors and a lower bound of the fraction of
        support vectors. You can think of it as the expected fraction of anomalies.
    optimizer
        The sequential optimizer used for updating the weights.
    intercept_lr
        Learning rate scheduler used for updating the intercept. A `optim.schedulers.Constant` is
        used if a `float` is provided. The intercept is not updated when this is set to 0.
    clip_gradient
        Clips the absolute value of each gradient value.
    initializer
        Weights initialization scheme.

    Examples
    --------

    >>> from river import anomaly
    >>> from river import compose
    >>> from river import datasets
    >>> from river import metrics
    >>> from river import preprocessing

    >>> model = anomaly.QuantileFilter(
    ...     anomaly.OneClassSVM(nu=0.2),
    ...     q=0.995
    ... )

    >>> auc = metrics.ROCAUC()

    >>> for x, y in datasets.CreditCard().take(2500):
    ...     score = model.score_one(x)
    ...     is_anomaly = model.classify(score)
    ...     model = model.learn_one(x)
    ...     auc = auc.update(y, is_anomaly)

    >>> auc
    ROCAUC: 74.68%

    You can also use the `evaluate.progressive_val_score` function to evaluate the model on a
    data stream.

    >>> from river import evaluate

    >>> model = model.clone()
    >>>

    >>> evaluate.progressive_val_score(
    ...     dataset=datasets.CreditCard().take(2500),
    ...     model=model,
    ...     metric=metrics.ROCAUC(),
    ...     print_every=1000
    ... )
    [1,000] ROCAUC: 74.40%
    [2,000] ROCAUC: 74.60%
    [2,500] ROCAUC: 74.68%
    ROCAUC: 74.68%

    """

    def __init__(
        self,
        nu=0.1,
        optimizer: optim.base.Optimizer | None = None,
        intercept_lr: optim.base.Scheduler | float = 0.01,
        clip_gradient=1e12,
        initializer: optim.base.Initializer | None = None,
    ):
        super().__init__(
            optimizer=optim.SGD(0.01) if optimizer is None else optimizer,
            loss=optim.losses.Hinge(),
            intercept_init=1.0,
            intercept_lr=intercept_lr,
            l2=nu / 2,
            l1=0,  # for compatibility, L1 here is not explicitly supported
            clip_gradient=clip_gradient,
            initializer=initializer if initializer else optim.initializers.Zeros(),
        )
        self.nu = nu

    def _get_intercept_update(self, loss_gradient):
        return (
            super()._get_intercept_update(loss_gradient)
            + 2.0 * self.intercept_lr.get(self.optimizer.n_iterations) * self.l2
        )

    def learn_one(self, x):
        return super().learn_one(x, y=1)

    def learn_many(self, X):
        return super().learn_many(X, y=pd.Series(True, index=X.index))

    def score_one(self, x):
        return self._raw_dot_one(x) - self.intercept
