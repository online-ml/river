# SMAPE

Symmetric mean absolute percentage error.

## Attributes

- **bigger\_is\_better**

  Indicate if a high value is better than a low one or not.
- **works\_with\_weights**

  Indicate whether the model takes into consideration the effect of sample weights

## Examples

```python
from river import metrics

y_true = [0, 0.07533, 0.07533, 0.07533, 0.07533, 0.07533, 0.07533, 0.0672, 0.0672]
y_pred = [0, 0.102, 0.107, 0.047, 0.1, 0.032, 0.047, 0.108, 0.089]

metric = metrics.SMAPE()
for yt, yp in zip(y_true, y_pred):
    metric.update(yt, yp)

metric
```

```text
SMAPE: 37.869392
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
> - **y\_true** — `float`
> - **y\_pred** — `float`
> - **w** — defaults to `1.0`

> [!NOTE]
>
> **update**
>
> Update the metric.
>
> **Parameters**
>
> - **y\_true** — `float`
> - **y\_pred** — `float`
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
