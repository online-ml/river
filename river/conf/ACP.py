from river import stats
from collections import deque
from typing import Tuple
import math

from conf.base import Interval


class AdaptativeConformalPrediction(Interval):
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

    >>> model = conf.AdaptativeConformalPrediction(
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
        self, window_size: int, gamma:float=0.5, alpha: float = 0.05,
    ):
        self.gamma = gamma
        self.alpha = alpha
        self.alpha_t = alpha
        self.window_size = window_size
        self.residuals = deque()

    def update(self, y_true: float, y_pred: float) -> "Interval":
        """Update the Interval."""

        if len(self.residuals)==self.window_size:
            # Remove the oldest residuals
            self.residuals.popleft()
            # Add the new one
            self.residuals.append(abs(y_true - y_pred))

            if(self.alpha_t >= 1): # => 1-alpha_t <= 0 => predict empty set
                err = 1 # err = 1 if the point is not included, 0 otherwise

            elif(self.alpha_t <= 0): # => 1-alpha_t >= 1 => predict the whole real line
                self.lower, self.upper = -math.inf, math.inf
                err = 0

            else: # => 1-alpha_t in ]0,1[ => compute the quantiles
                # Update the updated quantile
                rolling_quantile = stats.Quantile(1-self.alpha_t)
                for x in self.residuals:
                    _ = rolling_quantile .update(x)

                # Get the window
                half_inter = rolling_quantile .get()

                # create the bounds for the ACP interval, centered around y_pred
                self.lower, self.upper = y_pred-half_inter, y_pred+half_inter

                err = 1-float((self.lower <= y_true) & (y_true <= self.upper))

            # compute next value of alpha_t using updating scheme
            self.alpha_t = self.alpha_t + self.gamma*(self.alpha_t-err)

        else:
            # Fill the residuals until it reaches window size
            self.residuals.append(abs(y_true - y_pred))

    def get(self) -> Tuple[float, float]:
        """Return the current value of the Interval."""
        return (self.lower, self.upper)
