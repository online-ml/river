# BalancedAccuracy

Balanced accuracy.

Balanced accuracy is the average of recall obtained on each class. It is used to deal with imbalanced datasets in binary and multi-class classification problems.

## Parameters

- **cm**

  *Default* → `None`

  This parameter allows sharing the same confusion matrix between multiple metrics. Sharing a confusion matrix reduces the amount of storage and computation time.

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
y_true = [True, False, True, True, False, True]
y_pred = [True, False, True, True, True, False]

metric = metrics.BalancedAccuracy()
for yt, yp in zip(y_true, y_pred):
    metric.update(yt, yp)

metric
```

```text
BalancedAccuracy: 62.50%
```

```python
y_true = [0, 1, 0, 0, 1, 0]
y_pred = [0, 1, 0, 0, 0, 1]
metric = metrics.BalancedAccuracy()
for yt, yp in zip(y_true, y_pred):
    metric.update(yt, yp)

metric
```

```text
BalancedAccuracy: 62.50%
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
> - **y\_true**
> - **y\_pred**
> - **w** — defaults to `1.0`

> [!NOTE]
>
> **update**
>
> Update the metric.
>
> **Parameters**
>
> - **y\_true**
> - **y\_pred**
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
