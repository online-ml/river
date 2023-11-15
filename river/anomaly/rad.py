from __future__ import annotations

import math

from river import anomaly, base, linear_model, preprocessing, stats, time_series

__all__ = ["ReconstructionAnomalyDetecion"]


class ReconstructionAnomalyDetecion(anomaly.base.SupervisedAnomalyDetector):
    """Reconstruction Anomaly-Detection (RAD).
    This is the place for documentation
    """

    def __init__(
        self,
        reconstruction: base.Estimator | None = None,
        horizon: int | None = None,
        n_std: float | None = None,
        warmup_period: int | None = None,
    ):
        # Setting the reconstruction model that learns how normal data behaves
        self.reconstruction = (
            reconstruction
            if reconstruction is not None
            else preprocessing.MinMaxScaler() | linear_model.LinearRegression()
        )

        # Standard values for hyperparameters
        self.n_std = n_std if n_std is not None else 3
        self.horizon = horizon if horizon is not None else 1
        self.warmup_period = warmup_period if warmup_period is not None else 0

        # Initialize necessary stats
        self.dynamic_mean_squared_error = stats.Mean()
        self.dynamic_squared_error_variance = stats.Var()

        # Initialize necessary lists for tracking values
        self.predictions: list[float] = []
        self.squared_errors: list[float] = []
        self.thresholds: list[float] = []

        # Initialize necessary values for warm-up procedure
        self.iterations: int = 0
        self.warmed_up: bool = self.iterations >= self.warmup_period

    # This method is called to make the reconstruction model learn one example
    def learn_one(self, x: dict | None, y: base.typing.Target):
        self.iterations += 1
        self.warmed_up = self.iterations >= self.warmup_period

        # Checking if the features should be passed to model
        if x is None:
            self.reconstruction.learn_one(y)
        else:
            self.reconstruction.learn_one(x, y)
        return self

    # This method is calles to calculate an anomaly score for one example
    def score_one(self, x: dict, y: base.typing.Target):
        self.iterations += 1
        self.warmed_up = self.iterations >= self.warmup_period

        # Check if model is a time-series forecasting model
        if isinstance(self.reconstruction, time_series.base.Forecaster):
            y_pred = self.reconstruction.forecast(self.horizon)[0]
        else:
            y_pred = self.reconstruction.predict_one(x)
        self.predictions.append(y_pred)

        # Calculate the errors necessary for thresholding
        abs_error = abs(y_pred - y)
        squared_error = abs_error**2
        self.squared_errors.append(squared_error)

        # Based on the errors and hyperparameters, calculate threshold
        threshold = self.dynamic_mean_squared_error.get() + (
            self.n_std * math.sqrt(self.dynamic_squared_error_variance.get())
        )
        self.thresholds.append(threshold)

        self.dynamic_mean_squared_error.update(squared_error)
        self.dynamic_squared_error_variance.update(squared_error)

        # When warmup hyperparam is used, only return score if warmed up
        if not self.warmed_up:
            return 0.0

        # Every error above threshold is scored with 100% or 1.0
        # Everything below is distributed linearly from 0.0 - 0.999...
        if squared_error >= threshold:
            return 1.0
        else:
            return squared_error / threshold
