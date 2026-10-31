# FBeta

Binary F-Beta score.

The FBeta score is a weighted harmonic mean between precision and recall. The higher the `beta` value, the higher the recall will be taken into account. When `beta` equals 1, precision and recall and equivalently weighted, which results in the F1 score (see [`metrics.F1`](<https://riverml.xyz/dev/api/metrics/F1>)).

## Parameters

- **beta**

  *Type* → `float`

  Weight of precision in the harmonic mean.
- **cm**

  *Default* → `None`

  This parameter allows sharing the same confusion matrix between multiple metrics. Sharing a confusion matrix reduces the amount of storage and computation time.
- **pos\_val**

  *Default* → `True`

  Value to treat as "positive".

## Attributes

- **precision** ([`metrics.Precision`](<https://riverml.xyz/dev/api/metrics/Precision>))
- **recall** ([`metrics.Recall`](<https://riverml.xyz/dev/api/metrics/Recall>))

## Examples

```python
from river import metrics

y_true = [False, False, False, True, True, True]
y_pred = [False, False, True, True, False, False]

metric = metrics.FBeta(beta=2)
for yt, yp in zip(y_true, y_pred):
    metric.update(yt, yp)

metric
```

```text
FBeta: 35.71%
```

## Methods

> [!NOTE]
>
> **get**
>
> Return the current value of the metric.

> [!NOTE]
>
> **is\_better\_than**
>
> Indicate if the current metric is better than another one.
>
> **Parameters**
>
> - **other**

> [!NOTE]
>
> **revert**
>
> Revert the metric.
>
> **Parameters**
>
> - **y\_true** — `bool`
> - **y\_pred** — `bool | float | dict[bool, float]`
> - **w** — defaults to `1.0`

> [!NOTE]
>
> **update**
>
> Update the metric.
>
> **Parameters**
>
> - **y\_true** — `bool`
> - **y\_pred** — `bool | float | dict[bool, float]`
> - **w** — defaults to `1.0`

> [!NOTE]
>
> **works\_with**
>
> Indicates whether or not a metric can work with a given model.
>
> **Parameters**
>
> - **model** — `base.Estimator`
