# HoltWinters

Holt-Winters forecaster.

This is a standard implementation of the Holt-Winters forecasting method. Certain parametrisations result in special cases, such as simple exponential smoothing.

Optimal parameters and initialisation values can be determined in a batch setting. However, in an online setting, it is necessary to wait and observe enough values. The first `k = max(2, seasonality)` values are indeed used to initialize the components.

**Level initialization**

\\\[l = \\frac{1}{k} \\sum\_{i=1}{k} y\_i\\\]

**Trend initialization**

\\\[t = \\frac{1}{k - 1} \\sum\_{i=2}{k} y\_i - y\_{i-1}\\\]

**Trend initialization**

\\\[s\_i = \\frac{y\_i}{k}\\\]

## Parameters

- **alpha**

  Smoothing parameter for the level.
- **beta**

  *Default* → `None`

  Smoothing parameter for the trend.
- **gamma**

  *Default* → `None`

  Smoothing parameter for the seasonality.
- **seasonality**

  *Default* → `0`

  The number of periods in a season. For instance, this should be 4 for quarterly data, and 12 for yearly data.
- **multiplicative**

  *Default* → `False`

  Whether or not to use a multiplicative formulation.

## Examples

```python
from river import datasets
from river import evaluate
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

evaluate.evaluate(
    dataset,
    model,
    metric,
    horizon=12
)
```

```text
+1  MAE: 25.899087
+2  MAE: 26.26131
+3  MAE: 25.735903
+4  MAE: 25.625678
+5  MAE: 26.093842
+6  MAE: 26.90249
+7  MAE: 28.634398
+8  MAE: 29.284769
+9  MAE: 31.018351
+10 MAE: 32.252349
+11 MAE: 33.518946
+12 MAE: 33.975057
```

## Methods

> [!NOTE]
>
> **forecast**
>
> Makes forecast at each step of the given horizon.
>
> **Parameters**
>
> - **horizon** — `int`
> - **xs** — `list[dict] | None` — defaults to `None`

> [!NOTE]
>
> **learn\_one**
>
> Updates the model.
>
> **Parameters**
>
> - **y** — `float`
> - **x** — `dict | None` — defaults to `None`

## References

- [Exponential smoothing — Wikipedia](<https://en.wikipedia.org/wiki/Exponential_smoothing>)
- [Exponential smoothing — Forecasting: Principles and Practice](<https://otexts.com/fpp2/expsmooth.html>)
- [What is Exponential Smoothing? — Engineering statistics handbook](<https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc43.htm>)
