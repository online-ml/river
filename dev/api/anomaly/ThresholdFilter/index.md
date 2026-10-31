# ThresholdFilter

Threshold anomaly filter.

## Parameters

- **anomaly\_detector**

  An anomaly detector.
- **threshold**

  *Type* → `float`

  A threshold above which to classify an anomaly score as anomalous.
- **protect\_anomaly\_detector**

  *Default* → `True`

  Indicates whether or not the anomaly detector should be updated when the anomaly score is anomalous. If the data contains sporadic anomalies, then the anomaly detector should likely not be updated. Indeed, if it learns the anomaly score, then it will slowly start to consider anomalous anomaly scores as normal. This might be desirable, for instance in the case of drift.

## Examples

Anomaly filters can be used as part of a pipeline. For instance, we might want to filter out anomalous observations so as not to corrupt a supervised model. As an example, let's take the [`datasets.WaterFlow`](<https://riverml.xyz/dev/api/datasets/WaterFlow>) dataset. Some of the samples have anomalous target variables because of human interventions. We don't want our model to learn these values.

```python
from river import anomaly
from river import datasets
from river import evaluate
from river import metrics
from river import time_series

dataset = datasets.WaterFlow()
metric = metrics.SMAPE()

period = 24  # 24 samples per day

model = (
    anomaly.ThresholdFilter(
        anomaly.GaussianScorer(
            window_size=period * 7,  # 7 days
            grace_period=30
        ),
        threshold=0.995
    ) |
    time_series.HoltWinters(
        alpha=0.3,
        beta=0.1,
        multiplicative=False
    )
)

evaluate.evaluate(
    dataset,
    model,
    metric,
    horizon=period
)
```

```text
+1  SMAPE: 4.220171
+2  SMAPE: 4.322648
+3  SMAPE: 4.418546
+4  SMAPE: 4.504986
+5  SMAPE: 4.57924
+6  SMAPE: 4.64123
+7  SMAPE: 4.694042
+8  SMAPE: 4.740753
+9  SMAPE: 4.777291
+10 SMAPE: 4.804558
+11 SMAPE: 4.828114
+12 SMAPE: 4.849823
+13 SMAPE: 4.865871
+14 SMAPE: 4.871972
+15 SMAPE: 4.866274
+16 SMAPE: 4.842614
+17 SMAPE: 4.806214
+18 SMAPE: 4.763355
+19 SMAPE: 4.713455
+20 SMAPE: 4.672062
+21 SMAPE: 4.659102
+22 SMAPE: 4.693496
+23 SMAPE: 4.773707
+24 SMAPE: 4.880654
```

## Methods

> [!NOTE]
>
> **classify**
>
> Classify an anomaly score as anomalous or not.
>
> **Parameters**
>
> - **score** — `float`
>
> **Returns**
>
> `bool`: A boolean value indicating whether the anomaly score is anomalous or not.

> [!NOTE]
>
> **learn\_one**
>
> Update the anomaly filter and the underlying anomaly detector.
>
> **Parameters**
>
> - **args** — `Any`
> - **learn\_kwargs** — `Any`

> [!NOTE]
>
> **score\_one**
>
> Return an outlier score.
>
> A high score is indicative of an anomaly. A low score corresponds to a normal observation.
>
> **Parameters**
>
> - **args** — `Any`
> - **kwargs** — `Any`
>
> **Returns**
>
> `float`: An anomaly score. A high score is indicative of an anomaly. A low score corresponds a
