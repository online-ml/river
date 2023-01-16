from river import stats
from river import optim

from conf.base import Interval

from collections import deque
from typing import Tuple

class Gaussian(Interval):
    """Gaussian method to define intervals

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

    >>> model = conf.Gaussian(
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
    Mean: 0.903097

    The interval's efficiency is the average width of the intervals.

    >>> efficiency
    Mean: 3.430883

    Lowering the confidence will mechanically improve the efficiency.

    References
    ----------
    [^1]: [Barber, Rina Foygel, Emmanuel J. Candes, Aaditya Ramdas, and Ryan J. Tibshirani. "Predictive inference with the jackknife+." The Annals of Statistics 49, no. 1 (2021): 486-507.](https://www.stat.cmu.edu/~ryantibs/papers/jackknife.pdf)

    """

    def __init__(
        self, window_size: int, alpha: float = 0.05
    ):
        self.alpha = alpha
        self.window_size = window_size
        self.residuals = deque()
        self.var = stats.Var()

        # define the ppf on the normal distribution : 
        # Get a normal distribution sampler
        normal_sampler = optim.initializers.Normal(mu=0, sigma=1, seed=42)
        # And sample enough samples : 5 millions give the result of scipy
        normal_samples = normal_sampler(shape=500000)
        # Initialize a quantile computer in River
        rolling_quantile = stats.Quantile((1 - alpha) / 2)
        for x in normal_samples:
            _ = rolling_quantile.update(x)
        # set the result
        self.norm_ppf = rolling_quantile.get()

    def update(self, y_true: float, y_pred: float) -> "Interval":
        """Update the Interval."""
        
        if len(self.residuals)==self.window_size:
            # Remove the oldest residuals 
            self.residuals.popleft()
            # Add the new one
            self.residuals.append(y_true - y_pred)

            # Update the variance with the latest residual
            self.var.update(self.residuals[-1])      

            # Compute the interval
            # First get 
            half_inter = self.norm_ppf*self.var.get()**0.5
            # And set the borne
            self.lower, self.upper = y_pred-half_inter, y_pred+half_inter
        else:
            # Fill the residuals until it reaches window size
            self.residuals.append(y_true - y_pred)
            # Update the variance
            self.var.update(self.residuals[-1])

    def get(self) -> Tuple[float, float]:
        """Return the current value of the Interval."""
        return (self.lower, self.upper)

