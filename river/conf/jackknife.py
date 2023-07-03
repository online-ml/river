from __future__ import annotations

from river import base, stats

from . import interval


class RegressionJackknife(base.Wrapper, base.Regressor):
    """Jackknife method for regression.

    This is a conformal prediction method for regression. It is based on the jackknife method. The
    idea is to compute the quantiles of the residuals of the regressor. The prediction interval is
    then computed as the prediction of the regressor plus the quantiles of the residuals.

    This works naturally online, as the quantiles of the residuals are updated at each iteration.
    Each residual is produced before the regressor is updated, which ensures the predicted intervals
    are not optimistic.

    Note that the produced intervals are marginal and not conditional. This means that the intervals
    are not adjusted for the features `x`. This is a limitation of the jackknife method. However,
    the jackknife method is very simple and efficient. It is also very robust to outliers.

    Parameters
    ----------
    regressor
        The regressor to be wrapped.
    confidence_level
        The confidence level of the prediction intervals.
    window_size
        The size of the window used to compute the quantiles of the residuals. If `None`, the
        quantiles are computed over the whole history. It is advised to set this if you expect the
        model's performance to change over time.

    Examples
    --------

    >>> from river import conf
    >>> from river import datasets
    >>> from river import linear_model
    >>> from river import metrics
    >>> from river import preprocessing
    >>> from river import stats

    >>> dataset = datasets.TrumpApproval()

    >>> model = conf.RegressionJackknife(
    ...     (
    ...         preprocessing.StandardScaler() |
    ...         linear_model.LinearRegression(intercept_lr=.1)
    ...     ),
    ...     confidence_level=0.9
    ... )

    >>> validity = stats.Mean()
    >>> efficiency = stats.Mean()

    >>> for x, y in dataset:
    ...     interval = model.predict_one(x, with_interval=True)
    ...     validity = validity.update(y in interval)
    ...     efficiency = efficiency.update(interval.width)
    ...     model = model.learn_one(x, y)

    The interval's validity is the proportion of times the true value is within the interval. We
    specified a confidence level of 90%, so we expect the validity to be around 90%.

    >>> validity
    Mean: 0.939061

    The interval's efficiency is the average width of the intervals.

    >>> efficiency
    Mean: 4.078361

    Lowering the confidence lowering will mechanically improve the efficiency.

    References
    ----------
    [^1]: [Barber, Rina Foygel, Emmanuel J. Candes, Aaditya Ramdas, and Ryan J. Tibshirani. "Predictive inference with the jackknife+." The Annals of Statistics 49, no. 1 (2021): 486-507.](https://www.stat.cmu.edu/~ryantibs/papers/jackknife.pdf)

    """

    def __init__(
        self,
        regressor: base.Regressor,
        confidence_level: float = 0.95,
        window_size: int | None = None,
    ):
        self.regressor = regressor
        self.confidence_level = confidence_level
        self.window_size = window_size

        alpha = (1 - confidence_level) / 2
        self._lower = (
            stats.RollingQuantile(alpha, window_size) if window_size else stats.Quantile(alpha)
        )
        self._upper = (
            stats.RollingQuantile(1 - alpha, window_size)
            if window_size
            else stats.Quantile(1 - alpha)
        )

    @property
    def _wrapped_model(self):
        return self.regressor

    @classmethod
    def _unit_test_params(cls):
        from river import linear_model, preprocessing

        yield {"regressor": (preprocessing.StandardScaler() | linear_model.LinearRegression())}

    def learn_one(self, x, y, **kwargs):
        # Update the quantiles
        error = y - self.regressor.predict_one(x)
        self._lower.update(error)
        self._upper.update(error)

        self.regressor.learn_one(x, y, **kwargs)

        return self

    def predict_one(self, x, with_interval=False, **kwargs):
        """Predict the output of features `x`.

        Parameters
        ----------
        x
            A dictionary of features.
        with_interval
            Whether to return a predictive distribution, or instead just the most likely value.

        Returns
        -------
        The prediction.

        """
        y_pred = self.regressor.predict_one(x, **kwargs)

        if not with_interval:
            return y_pred

        return interval.Interval(
            lower=y_pred + (self._lower.get() or 0), upper=y_pred + (self._upper.get() or 0)
        )
