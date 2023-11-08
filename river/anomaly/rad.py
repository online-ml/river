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
        n_std: int | None = None,
    ):
        self.reconstruction = (
            reconstruction
            if reconstruction is not None
            else preprocessing.MinMaxScaler() | linear_model.LinearRegression()
        )

        self.n_std = n_std if n_std is not None else 3

        self.horizon = horizon if horizon is not None else 1

        self.dynamic_mean_squared_error = stats.Mean()
        self.dynamic_squared_error_variance = stats.Var()
        self.predictions: list[float] = []
        self.squared_errors: list[float] = []
        self.thresholds: list[float] = []

    def learn_one(self, x: dict, y: base.typing.Target):
        self.reconstruction.learn_one(x, y)
        return self

    def score_one(self, x: dict, y: base.typing.Target):
        if isinstance(self.reconstruction, time_series.base.Forecaster):
            y_pred = self.reconstruction.forecast(self.horizon)[0]
        else:
            y_pred = self.reconstruction.predict_one(x)
        self.predictions.append(y_pred)

        abs_error = abs(y_pred - y)
        squared_error = abs_error**2
        self.squared_errors.append(squared_error)

        threshold = self.dynamic_mean_squared_error.get() + (
            self.n_std * math.sqrt(self.dynamic_squared_error_variance.get())
        )
        self.thresholds.append(threshold)

        self.dynamic_mean_squared_error.update(squared_error)
        self.dynamic_squared_error_variance.update(squared_error)

        self.reconstruction.learn_one(x, y)

        if squared_error >= threshold:
            return 1.0
        else:
            return squared_error / threshold
