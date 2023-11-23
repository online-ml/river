from __future__ import annotations

import math

from river import anomaly, base, linear_model, preprocessing, stats, time_series

__all__ = ["PredictiveAnomalyDetection"]


class PredictiveAnomalyDetection(anomaly.base.SupervisedAnomalyDetector):
    """Predictive Anomaly Detection.

    This semi-supervised technique to anomaly detection employs a predictive model to learn the normal behavior
    of a dataset. It forecasts future data points and compares these predictions with actual values to determine
    anomalies. An anomaly score is generated based on the deviation of the prediction from the actual value, with higher
    scores indicating a higher probability of an anomaly.

    The actual anomaly-score is calculated by comparing the squared-error to a dynamic threshold. If the error larger
    than this threshold, the score will be 1.0. If the error is smaller, then the score will be linearly distributed
    between 0.0 and 0.999... depending on the distance to the threshold.

    Parameters
    ----------
    predictive_model
        The underlying model that learns the normal behavior of the data and makes predictions on future behavior.
        This can be any Estimator, depending on the type of problem (e.g. some Forecaster for Time-Series Data).
    horizon
        When using a Forecaster this is the horizon of its forecasts.
    n_std
        Number of Standard Deviations to use for calculating the threshold. Larger numbers will result in the model
        being less sensitive.
    warmup_period
        Number of iterations for the model to warm up. Since the model will start with no knowledge,
        the first predictions will be bad resulting in a high error (which is normal).
        Therefore the first instance will have a very high anomaly-score. While the model is warming up,
        no score will be calculated and the score_one method will just return 0.

    Attributes
    ----------
    dynamic_mean_squared_error : stats.Mean
         The Running mean of the (squared) errors the model made to update the dynamic threshold.
    dynamic_squared_error_variance : stats.Var
        The running variance of the (squared) errors the model made to update the dynamic threshold.
    iterations : int
        The number of iterations the model has seen

    Examples
    --------

    >>> from river import datasets
    >>> from river import time_series
    >>> from river import anomaly
    >>> from river import preprocessing
    >>> from river import linear_model
    >>> from river import optim

    >>> period = 12
    >>> predictive_model = time_series.SNARIMAX(
    ...     p=period,
    ...     d=1,
    ...     q=period,
    ...     m=period,
    ...     sd=1,
    ...     regressor=(
    ...         preprocessing.StandardScaler()
    ...         | linear_model.LinearRegression(
    ...             optimizer=optim.SGD(0.005),
    ...         )
    ...     ),
    ... )

    >>> PAD = anomaly.PredictiveAnomalyDetection(
    ...     predictive_model,
    ...     horizon=1,
    ...     n_std=3.5,
    ...     warmup_period=15
    ... )

    >>> for t, (x, y) in enumerate(datasets.AirlinePassengers()):
    ...     score = PAD.score_one(None, y)
    ...     PAD.learn_one(None, y)
    ...     print(score)
    0.0
    0.0
    0.0
    0.0
    ...
    5.477831890312668e-05
    0.07305562392710468
    0.030122505497227493
    0.04803795404401492
    0.014216675596576562
    0.04789677144570603
    0.003410489566495498


    References
    ----------
    [^1]: [Generic and Scalable Framework for Automated Time-series Anomaly Detection](https://dl.acm.org/doi/abs/10.1145/2783258.2788611)
    """

    def __init__(
        self,
        predictive_model: base.Estimator | None = None,
        horizon: int = 1,
        n_std: float = 3.0,
        warmup_period: int = 0,
    ):
        # Setting the predictive model that learns how normal data behaves
        self.predictive_model = (
            predictive_model
            if predictive_model is not None
            else preprocessing.MinMaxScaler() | linear_model.LinearRegression()
        )

        self.horizon = horizon
        self.n_std = n_std
        self.warmup_period = warmup_period

        # Initialize necessary statistical measures
        self.dynamic_mean_squared_error = stats.Mean()
        self.dynamic_squared_error_variance = stats.Var()

        # Initialize necessary values for warm-up procedure
        self.iterations: int = 0

    # This method is called to make the predictive model learn one example
    def learn_one(self, x: dict | None, y: base.typing.Target | float):
        self.iterations += 1

        # Check if model is a time-series forecasting model or regressor/classification
        if isinstance(self.predictive_model, time_series.base.Forecaster):
            # When theres no feature-dict just pass target to forecaster
            if not x:
                self.predictive_model.learn_one(y)
            else:
                self.predictive_model.learn_one(y, x)
        else:
            self.predictive_model.learn_one(x=x, y=y)
        return self

    # This method is calles to calculate an anomaly score for one example
    def score_one(self, x: dict, y: base.typing.Target):
        # Check if model is a time-series forecasting model
        if isinstance(self.predictive_model, time_series.base.Forecaster):
            y_pred = self.predictive_model.forecast(self.horizon)[0]
        else:
            y_pred = self.predictive_model.predict_one(x)

        # Calculate the errors necessary for thresholding
        squared_error = (y_pred - y) ** 2

        # Based on the errors and hyperparameters, calculate threshold
        threshold = self.dynamic_mean_squared_error.get() + (
            self.n_std * math.sqrt(self.dynamic_squared_error_variance.get())
        )

        self.dynamic_mean_squared_error.update(squared_error)
        self.dynamic_squared_error_variance.update(squared_error)

        # When warmup hyperparam is used, only return score if warmed up
        if self.iterations < self.warmup_period:
            return 0.0

        # Every error above threshold is scored with 100% or 1.0
        # Everything below is distributed linearly from 0.0 - 0.999...
        if squared_error >= threshold:
            return 1.0
        else:
            return squared_error / threshold

    # This version of score_one also returns the score along with the prediction, error and threshold of the model
    def score_one_detailed(
        self, x: dict, y: base.typing.Target
    ) -> tuple[float, base.typing.Target, float, float]:
        if isinstance(self.predictive_model, time_series.base.Forecaster):
            y_pred = self.predictive_model.forecast(self.horizon)[0]
        else:
            y_pred = self.predictive_model.predict_one(x)

        squared_error = (y_pred - y) ** 2

        threshold = self.dynamic_mean_squared_error.get() + (
            self.n_std * math.sqrt(self.dynamic_squared_error_variance.get())
        )

        self.dynamic_mean_squared_error.update(squared_error)
        self.dynamic_squared_error_variance.update(squared_error)

        score: float = 0.0

        if self.iterations < self.warmup_period:
            score = 0.0
        else:
            if squared_error >= threshold:
                score = 1.0
            else:
                score = squared_error / threshold

        return (score, y_pred, squared_error, threshold)
