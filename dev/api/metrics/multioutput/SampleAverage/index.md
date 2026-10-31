# SampleAverage

Sample-average wrapper.

The provided metric is evaluate on each sample. The arithmetic average over all the samples is returned. This is equivalent to using `average='samples'` in scikit-learn.

## Parameters

- **metric**

  A classification or a regression metric.

## Attributes

- **bigger\_is\_better**

  Indicate if a high value is better than a low one or not.
- **metric**

  Gives access to the wrapped metric.
- **requires\_labels**
- **works\_with\_weights**

  Indicate whether the model takes into consideration the effect of sample weights

## Examples

```python
from river import metrics

y_true = [
    {0: False, 1: True, 2: True},
    {0: True, 1: True, 2: False}
]
y_pred = [
    {0: True, 1: True, 2: True},
    {0: True, 1: False, 2: False}
]

sample_jaccard = metrics.multioutput.SampleAverage(metrics.Jaccard())

for yt, yp in zip(y_true, y_pred):
    sample_jaccard.update(yt, yp)

sample_jaccard
```

```text
SampleAverage(Jaccard): 58.33%
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
