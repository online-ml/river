# StandardAbsoluteDeviation

Standard Absolute Deviation (SAD).

SAD is the model that calculates the anomaly score by using the deviation from the mean/median, divided by the standard deviation of all the points seen within the data stream. The idea of this model is based on the \\(3 \\times \\sigma\\) rule described in [^/dev/api/anomaly/StandardAbsoluteDeviation/#1].

This implementation is adapted from the [implementation](<https://github.com/selimfirat/pysad/blob/master/pysad/models/standard_absolute_deviation.py>) within PySAD (Python Streaming Anomaly Detection) [^/dev/api/anomaly/StandardAbsoluteDeviation/#2].

As a univariate anomaly detection algorithm, this implementation is adapted to `River` in a similar way as that of the `GaussianScorer` algorithm, with the variable taken into the account at the learning phase and scoring phase under variable `y`, ignoring `x`.

## Parameters

- **sub\_stat**

  *Type* → `stats.base.Univariate | None`

  *Default* → `None`

  The statistic to be subtracted, then divided by the standard deviation for scoring. Defaults to `stats.Mean()`.

## Examples

```python
import random
from river import anomaly
from river import stats
from river import stream

rng = random.Random(42)

model = anomaly.StandardAbsoluteDeviation(sub_stat=stats.Mean())

for _ in range(150):
    y = rng.gauss(0, 1)
    model.learn_one(None, y)

model.score_one(None, 2)
```

```text
2.057...
```

```python
model.score_one(None, 0)
```

```text
0.084...
```

```python
model.score_one(None, 1)
```

```text
0.986...
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
> - **x** — `dict[base.typing.FeatureName, Any]`
> - **y** — `base.typing.Target`

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
> - **x** — `dict[base.typing.FeatureName, Any]`
> - **y** — `base.typing.Target`
>
> **Returns**
>
> `float`: An anomaly score. A high score is indicative of an anomaly. A low score corresponds a

[^/dev/api/anomaly/StandardAbsoluteDeviation/#1]: Hochenbaum, J., Vallis, O.S., Kejariwal, A., 2017. Automatic Anomaly Detection in the Cloud Via Statistical Learning. https://doi.org/10.48550/ARXIV.1704.07706.

[^/dev/api/anomaly/StandardAbsoluteDeviation/#2]: Yilmaz, S.F., Kozat, S.S., 2020. PySAD: A Streaming Anomaly Detection Framework in Python. https://doi.org/10.48550/ARXIV.2009.02572.
