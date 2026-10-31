# HorizonAggMetric

Same as `HorizonMetric`, but aggregates the result based on an provided function.

This allows, for instance, to measure the average performance of a forecasting model along the horizon.

## Parameters

- **metric**

  *Type* → `metrics.base.RegressionMetric`

  A regression metric.
- **agg\_func**

  *Type* → `typing.Callable[[list[float]], float]`

  A function that takes as input a list of floats and outputs a single float. You may want to `min`, `max`, as well as `statistics.mean` and `statistics.median`.

## Examples

This is used internally by the [`evaluate.evaluate`](<https://riverml.xyz/dev/api/evaluate/evaluate>) function when you pass an `agg_func`.

```python
import statistics
from river import evaluate
from river import datasets
from river import metrics
from river import time_series

metric = evaluate.evaluate(
    dataset=datasets.AirlinePassengers(),
    model=time_series.HoltWinters(alpha=0.1),
    metric=metrics.MAE(),
    agg_func=statistics.mean,
    horizon=4
)

metric
```

```text
mean(MAE): 42.901748
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
