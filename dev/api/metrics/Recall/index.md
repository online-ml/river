# Recall

Binary recall score.

## Parameters

- **cm**

  *Default* → `None`

  This parameter allows sharing the same confusion matrix between multiple metrics. Sharing a confusion matrix reduces the amount of storage and computation time.
- **pos\_val**

  *Default* → `True`

  Value to treat as "positive".

## Attributes

- **bigger\_is\_better**

  Indicate if a high value is better than a low one or not.
- **requires\_labels**

  Indicates if labels are required, rather than probabilities.
- **works\_with\_weights**

  Indicate whether the model takes into consideration the effect of sample weights

## Examples

```python
from river import metrics

y_true = [True, False, True, True, True]
y_pred = [True, True, False, True, True]

metric = metrics.Recall()

for yt, yp in zip(y_true, y_pred):
    metric.update(yt, yp)
    print(metric)
```

```text
Recall: 100.00%
Recall: 100.00%
Recall: 50.00%
Recall: 66.67%
Recall: 75.00%
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
