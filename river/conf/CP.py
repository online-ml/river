from river import stats
from collections import deque
from typing import Tuple

from conf.base import Interval


class ConformalPrediction(Interval):
    """Adapatative Conformal Prediction method

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

    >>> model = conf.ConformalPrediction(
    ...     (
    ...         preprocessing.StandardScaler() |
    ...         linear_model.LinearRegression(intercept_lr=.1)
    ...     ),
    ...     confidence_level=0.9,
            gamma=.05
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
    Mean: 0.903097

    The interval's efficiency is the average width of the intervals.

    >>> efficiency
    Mean: 3.430883

    Lowering the confidence lowering will mechanically improve the efficiency.

    References
    ----------
    [^1]: [Margaux Zaffran, Olivier Féron, Yannig Goude, Julie Josse, Aymeric Dieuleveut.
    "Adaptive Conformal Predictions for Time Series](https://arxiv.org/abs/2202.07282)

    """

    def __init__(
        self, window_size: int, alpha: float = 0.05
    ):
        self.alpha = alpha
        self.window_size = window_size
        self.residuals = deque()
        self.rolling_quantile = stats.Quantile((1-self.alpha)*(1+1/self.window_size))

    def update(self, y_true: float, y_pred: float) -> "Interval":
        """Update the Interval."""

        if len(self.residuals)==self.window_size:
            # Remove the oldest residuals
            self.residuals.popleft()
            # Add the new one
            self.residuals.append(abs(y_true - y_pred))

            # Update the quantile
            _ = self.rolling_quantile.update(self.residuals[-1])

            # Compute the interval
            half_inter = self.rolling_quantile.get()

            # And set the borne
            self.lower, self.upper = y_pred-half_inter, y_pred+half_inter
        else:
            # Fill the residuals until it reaches window size
            self.residuals.append(abs(y_true - y_pred))

    def get(self) -> Tuple[float, float]:
        """Return the current value of the Interval."""
        return (self.lower, self.upper)
