from __future__ import annotations

import math

from river import anomaly, base, linear_model, preprocessing, stats, time_series

__all__ = ["PredictiveAnomalyDetection"]


class PredictiveAnomalyDetection(anomaly.base.SupervisedAnomalyDetector):
    """Predictive Anomaly Detection.

    This semi-supervised technique to anomaly detection employs a predictive model to learn the normal behavior
    of a dataset. It forecasts future data points and compares these predictions with actual values to determine
    anomalies. An anomaly score is calculated based on the deviation of the prediction from the actual value, with higher
    scores indicating a higher probability of an anomaly.

    The actual anomaly score is calculated by comparing the squared-error to a dynamic threshold. If the error is larger
    than this threshold, the score will be 1.0; else, the score will be linearly distributed within the range (0.0, 1.0),
    with a higher score indicating a higher squared error compared to the threshold.

    Parameters
    ----------
    predictive_model
        The underlying model that learns the normal behavior of the data and makes predictions on future behavior.
        This can be an estimator of any type, depending on the type of problem (e.g. some Forecaster for Time-Series Data).
    horizon
        When a Forecaster is used as a predictive model, this is the horizon of its forecasts.
    n_std
        Number of Standard Deviations to calculate the threshold. A larger number of standard deviation will result in
        a higher threshold, resulting in the model being less sensitive.
    warmup_period
        Duration for the model to warm up. Since the model starts with zero knowledge,
        the first instances will have very high anomaly scores, resulting in bad predictions (or high error). As such,
        a warm-up period is necessary to discard the first seen instances.
        While the model is within the warm-up period, no score will be calculated and the score_one method will return 0.0.

    Attributes
    ----------
    dynamic_mae : stats.Mean
         The running mean of the (squared) errors from the predictions of the model to update the dynamic threshold.
    dynamic_se_variance : stats.Var
        The running variance of the (squared) errors from the predictions of the model to update the dynamic threshold.
    iter : int
        The number of iterations (data points) passed.

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

    >>> scores = []

    >>> for t, (x, y) in enumerate(datasets.AirlinePassengers()):
    ...     score = PAD.score_one(None, y)
    ...     PAD.learn_one(None, y)
    ...     scores.append(score)

    >>> print(scores[-1])
    0.05329236123455621

    References
    ----------
    [^1]: Laptev N, Amizadeh S, Flint I. Generic and scalable framework for Automated Time-series Anomaly Detection.
    Proceedings of the 21st ACM SIGKDD International Conference on Knowledge Discovery and Data Mining 2015.
    doi:10.1145/2783258.2788611.
    """

    def __init__(
        self,
        predictive_model: base.Estimator | None = None,
        horizon: int = 1,
        n_std: float = 3.0,
        warmup_period: int = 0,
    ):
        self.predictive_model = (
            predictive_model
            if predictive_model is not None
            else preprocessing.MinMaxScaler() | linear_model.LinearRegression()
        )

        self.horizon = horizon
        self.n_std = n_std
        self.warmup_period = warmup_period

        # Initialize necessary statistical measures
        self.dynamic_mae = stats.Mean()
        self.dynamic_se_variance = stats.Var()

        # Initialize necessary values for warm-up procedure
        self.iter: int = 0

    # This method is called to make the predictive model learn one example
    def learn_one(self, x: dict | None, y: base.typing.Target | float):
        self.iter += 1

        # Check whether the model is a time-series forecasting or regression/classification model
        if isinstance(self.predictive_model, time_series.base.Forecaster) and isinstance(y, float):
            # When there's no data point as dict of features, the target will be passed
            # to the forecaster as an exogenous variable.
            if not x:
                self.predictive_model.learn_one(y=y)
            else:
                self.predictive_model.learn_one(y=y, x=x)
        else:
            self.predictive_model.learn_one(x=x, y=y)

    def score_one(self, x: dict, y: base.typing.Target):
        # Return the predicted value of x from the predictive model, first by checking whether
        # it is a time-series forecaster.
        if isinstance(self.predictive_model, time_series.base.Forecaster):
            y_pred = self.predictive_model.forecast(self.horizon)[0]
        else:
            y_pred = self.predictive_model.predict_one(x)

        # Calculate the squared error
        squared_error = (y_pred - y) ** 2

        # Calculate the threshold
        threshold = self.dynamic_mae.get() + (
            self.n_std * math.sqrt(self.dynamic_se_variance.get())
        )

        # When warmup hyper-parameter is used, the anomaly score is only returned once the warmup period has passed.
        # When the warmup period has not passed, the default value of the anomaly score is 0.0
        if self.iter < self.warmup_period:
            return 0.0

        # Update MAE and SEV when the warm-up parameter has passed.
        self.dynamic_mae.update(squared_error)
        self.dynamic_se_variance.update(squared_error)

        # An error above the threshold will result in a score of 1.0.
        # Else, the score will be linearly distributed within the interval (0.0, 1.0)
        if squared_error >= threshold:
            return 1.0
        else:
            return squared_error / threshold
