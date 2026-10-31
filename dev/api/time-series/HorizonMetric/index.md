# HorizonMetric

Measures performance at each time step ahead.

This allows to measure the performance of a model at each time step along the horizon. A copy of the provided regression metric is made for each time step. At each time step ahead, the metric is thus evaluated on each prediction for said time step, and not for the time steps before or after that.

## Parameters

- **metric**

  *Type* → `metrics.base.RegressionMetric`

  A regression metric.

## Examples

This is used internally by the [`evaluate.evaluate`](<https://riverml.xyz/dev/api/evaluate/evaluate>) function.

```python
from river import evaluate
from river import datasets
from river import metrics
from river import time_series

metric = evaluate.evaluate(
    dataset=datasets.AirlinePassengers(),
    model=time_series.HoltWinters(alpha=0.1),
    metric=metrics.MAE(),
    horizon=4
)

metric
```

```text
+1 MAE: 40.931286
+2 MAE: 42.667998
+3 MAE: 44.158092
+4 MAE: 43.849617
```

## Methods

> [!NOTE]
>
> **get**
>
> Return the current performance along the horizon.
>
> **Returns**
>
> `list&#91;float&#93;`: The current performance.

> [!NOTE]
>
> **update**
>
> Update the metric at each step along the horizon.
>
> **Parameters**
>
> - **y\_true** — `list[Number]`
> - **y\_pred** — `list[Number]`
