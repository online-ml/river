# ExactMatch

Exact match score.

This is the most strict multi-label metric, defined as the number of samples that have all their labels correctly classified, divided by the total number of samples.

## Attributes

- **bigger\_is\_better**

  Indicate if a high value is better than a low one or not.
- **requires\_labels**
- **works\_with\_weights**

  Indicate whether the model takes into consideration the effect of sample weights

## Examples

```python
from river import metrics

y_true = [
    {0: False, 1: True, 2: True},
    {0: True, 1: True, 2: False},
    {0: True, 1: True, 2: False},
]

y_pred = [
    {0: True, 1: True, 2: True},
    {0: True, 1: False, 2: False},
    {0: True, 1: True, 2: False},
]

metric = metrics.multioutput.ExactMatch()
for yt, yp in zip(y_true, y_pred):
    metric.update(yt, yp)

metric
```

```text
ExactMatch: 33.33%
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
> - **y\_true** — `dict[str | int, base.typing.ClfTarget]`
> - **y\_pred** — `dict[str | int, base.typing.ClfTarget] | dict[str | int, dict[base.typing.ClfTarget, float]]`
> - **w** — defaults to `1.0`

> [!NOTE]
>
> **update**
>
> Update the metric.
>
> **Parameters**
>
> - **y\_true** — `dict[str | int, base.typing.ClfTarget]`
> - **y\_pred** — `dict[str | int, base.typing.ClfTarget] | dict[str | int, dict[base.typing.ClfTarget, float]]`
> - **w** — defaults to `1.0`

> [!NOTE]
>
> **works\_with**
>
> Indicates whether or not a metric can work with a given model.
>
> **Parameters**
>
> - **model**
