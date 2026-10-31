# MicroFBeta

Micro-average F-Beta score.

This computes the F-Beta score by merging all the predictions and true labels, and then computes a global F-Beta score.

## Parameters

- **beta**

  *Type* → `float`

  Weight of precision in the harmonic mean.
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

y_true = [0, 1, 2, 2, 0]
y_pred = [0, 1, 1, 2, 1]

metric = metrics.MicroFBeta(beta=2)
for yt, yp in zip(y_true, y_pred):
    metric.update(yt, yp)

metric
```

```text
MicroFBeta: 60.00%
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

## References

- 1. [Why are precision, recall and F1 score equal when using micro averaging in a multi-class problem?](<https://simonhessner.de/why-are-precision-recall-and-f1-score-equal-when-using-micro-averaging-in-a-multi-class-problem/>)
