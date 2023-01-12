from river import datasets
from river import metrics
from .conf.gaussian import Gaussian
from .evaluate import evaluate
from .holt_winters import HoltWinters

dataset = datasets.AirlinePassengers()

model = HoltWinters(
        alpha=0.3,
        beta=0.1,
        gamma=0.6,
        seasonality=12,
        multiplicative=True
        )

calib_period = 100
metric = metrics.MAE()
interval = Gaussian(window_size=calib_period)

evaluate(
        dataset,
        model,
        metric,
        interval,
        horizon=12,
        residual_calibration_period = calib_period
    )