# QuantileFilter

Threshold anomaly filter.

## Parameters

- **anomaly\_detector**

  An anomaly detector.
- **q**

  *Type* → `float`

  The quantile level above which to classify an anomaly score as anomalous.
- **protect\_anomaly\_detector**

  *Default* → `True`

  Indicates whether or not the anomaly detector should be updated when the anomaly score is anomalous. If the data contains sporadic anomalies, then the anomaly detector should likely not be updated. Indeed, if it learns the anomaly score, then it will slowly start to consider anomalous anomaly scores as normal. This might be desirable, for instance in the case of drift.

## Attributes

- **q**

## Examples

```python
from river import anomaly
from river import compose
from river import datasets
from river import metrics
from river import preprocessing

model = compose.Pipeline(
    preprocessing.MinMaxScaler(),
    anomaly.QuantileFilter(
        anomaly.HalfSpaceTrees(seed=42),
        q=0.95
    )
)

report = metrics.ClassificationReport()

for x, y in datasets.CreditCard().take(2000):
    score = model.score_one(x)
    is_anomaly = model['QuantileFilter'].classify(score)
    model.learn_one(x)
    report.update(y, is_anomaly)

report
```

```text
               Precision   Recall   F1       Support
<BLANKLINE>
       0      99.95%   94.49%   97.14%      1998
       1       0.90%   50.00%    1.77%         2
<BLANKLINE>
   Macro      50.42%   72.25%   49.46%
   Micro      94.45%   94.45%   94.45%
Weighted      99.85%   94.45%   97.05%
<BLANKLINE>
                 94.45% accuracy
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
