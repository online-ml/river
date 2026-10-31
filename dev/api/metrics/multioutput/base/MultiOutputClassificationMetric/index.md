# MultiOutputClassificationMetric

Mother class for all multi-output classification metrics.

## Parameters

- **cm**

  *Type* → `MultiLabelConfusionMatrix | None`

  *Default* → `None`

  This parameter allows sharing the same confusion matrix between multiple metrics. Sharing a confusion matrix reduces the amount of storage and computation time.

## Attributes

- **bigger\_is\_better**

  Indicate if a high value is better than a low one or not.
- **requires\_labels**
- **works\_with\_weights**

  Indicate whether the model takes into consideration the effect of sample weights

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
> - **model** — `base.Estimator`
