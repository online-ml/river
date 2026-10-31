# PredictiveAnomalyDetection

Predictive Anomaly Detection.

This semi-supervised technique to anomaly detection employs a predictive model to learn the normal behavior of a dataset. It forecasts future data points and compares these predictions with actual values to determine anomalies. An anomaly score is calculated based on the deviation of the prediction from the actual value, with higher scores indicating a higher probability of an anomaly.

The actual anomaly score is calculated by comparing the squared-error to a dynamic threshold. If the error is larger than this threshold, the score will be 1.0; else, the score will be linearly distributed within the range (0.0, 1.0), with a higher score indicating a higher squared error compared to the threshold.

## Parameters

- **predictive\_model**

  *Type* → `base.Estimator | None`

  *Default* → `None`

  The underlying model that learns the normal behavior of the data and makes predictions on future behavior. This can be an estimator of any type, depending on the type of problem (e.g. some Forecaster for Time-Series Data).
- **horizon**

  *Type* → `int`

  *Default* → `1`

  When a Forecaster is used as a predictive model, this is the horizon of its forecasts.
- **n\_std**

  *Type* → `float`

  *Default* → `3.0`

  Number of Standard Deviations to calculate the threshold. A larger number of standard deviation will result in a higher threshold, resulting in the model being less sensitive.
- **warmup\_period**

  *Type* → `int`

  *Default* → `0`

  Duration for the model to warm up. Since the model starts with zero knowledge, the first instances will have very high anomaly scores, resulting in bad predictions (or high error). As such, a warm-up period is necessary to discard the first seen instances. While the model is within the warm-up period, no score will be calculated and the score\_one method will return 0.0.

## Attributes

- **dynamic\_mae** ([`stats.Mean`](<https://riverml.xyz/dev/api/stats/Mean>))

  The running mean of the (squared) errors from the predictions of the model to update the dynamic threshold.
- **dynamic\_se\_variance** ([`stats.Var`](<https://riverml.xyz/dev/api/stats/Var>))

  The running variance of the (squared) errors from the predictions of the model to update the dynamic threshold.
- **iter** (`int`)

  The number of iterations (data points) passed.

## Examples

```python
from river import datasets
from river import time_series
from river import anomaly
from river import preprocessing
from river import linear_model
from river import optim

period = 12
predictive_model = time_series.SNARIMAX(
    p=period,
    d=1,
    q=period,
    m=period,
    sd=1,
    regressor=(
        preprocessing.StandardScaler()
        | linear_model.LinearRegression(
            optimizer=optim.SGD(0.005),
        )
    ),
)

PAD = anomaly.PredictiveAnomalyDetection(
    predictive_model,
    horizon=1,
    n_std=3.5,
    warmup_period=15
)

scores = []

for t, (x, y) in enumerate(datasets.AirlinePassengers()):
    score = PAD.score_one(None, y)
    PAD.learn_one(None, y)
    scores.append(score)

print(scores[-1])
```

```text
0.0532923612345562
```

## Methods

> [!NOTE]
>
> **learn\_one**
>
> Update the model.
>
> **Parameters**
>
> - **x** — `dict | None`
> - **y** — `base.typing.Target | float`

> [!NOTE]
>
> **score\_one**
>
> Return an outlier score.
>
> A high score is indicative of an anomaly. A low score corresponds a normal observation.
>
> **Parameters**
>
> - **x** — `dict`
> - **y** — `base.typing.Target`
>
> **Returns**
>
> `float`: An anomaly score. A high score is indicative of an anomaly. A low score corresponds a

## References

- Laptev N, Amizadeh S, Flint I. Generic and scalable framework for Automated Time-series Anomaly Detection. Proceedings of the 21st ACM SIGKDD International Conference on Knowledge Discovery and Data Mining 2015. doi:10.1145/2783258.2788611.
