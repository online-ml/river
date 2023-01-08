from river import datasets
from river import metrics
from river import time_series

dataset = datasets.AirlinePassengers()

model = time_series.HoltWinters(
        alpha=0.3,
        beta=0.1,
        gamma=0.6,
        seasonality=12,
        multiplicative=True
        )

metric = metrics.MAE()

time_series.evaluate(
        dataset,
        model,
        metric,
        horizon=12
    )