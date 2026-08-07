from __future__ import annotations

from river import linear_model, optim

__all__ = ["QuantileRegressor"]


class QuantileRegressor(linear_model.LinearRegression):
    """Linear regression model for conditional quantiles.

    The default `quantile=0.5` gives median regression. Higher or lower quantiles can be useful
    when the target distribution is asymmetric.

    Parameters
    ----------
    quantile
        The quantile to estimate. Values must be strictly between 0 and 1.
    optimizer
        Optimizer used for the weights.
    l2
        L2 regularization amount.
    l1
        L1 regularization amount.
    intercept_init
        Initial intercept value.
    intercept_lr
        Learning rate used for the intercept.
    clip_gradient
        Gradient clipping threshold.
    initializer
        Weight initializer.

    Examples
    --------

    >>> from river import linear_model
    >>> from river import optim

    >>> model = linear_model.QuantileRegressor(
    ...     quantile=0.5,
    ...     intercept_lr=optim.schedulers.InverseScaling(0.4, power=0.5),
    ... )

    >>> for y in [1, 2, 2, 3, 100]:
    ...     model.learn_one({}, y)

    >>> model.predict_one({})
    0.646...

    References
    ----------
    [^1]: [Scikit-learn documentation on QuantileRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.QuantileRegressor.html)

    """

    def __init__(
        self,
        quantile: float = 0.5,
        optimizer: optim.base.Optimizer | None = None,
        l2=0.0,
        l1=0.0,
        intercept_init=0.0,
        intercept_lr: optim.base.Scheduler | float = 0.01,
        clip_gradient=1e12,
        initializer: optim.base.Initializer | None = None,
    ):
        if not 0 < quantile < 1:
            raise ValueError("quantile must be strictly between 0 and 1")

        super().__init__(
            optimizer=optimizer,
            loss=optim.losses.Quantile(quantile),
            l2=l2,
            l1=l1,
            intercept_init=intercept_init,
            intercept_lr=intercept_lr,
            clip_gradient=clip_gradient,
            initializer=initializer,
        )

    @property
    def quantile(self):
        return self.loss.alpha

    @property
    def _mutable_attributes(self):
        return super()._mutable_attributes - {"loss"}
