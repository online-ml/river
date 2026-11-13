# ModelSelectionRegressor

A model selector for regression.

## Parameters

- **models**

  *Type* → `Iterator[base.Estimator]`
- **metric**

  *Type* → `metrics.base.Metric`

## Attributes

- **best\_model**

  The current best model.
- **models**

## Methods

> [!NOTE]
>
> **learn\_one**
>
> Fits to a set of features `x` and a real-valued target `y`.
>
> **Parameters**
>
> - **x** — `dict[base.typing.FeatureName, Any]`
> - **y** — `base.typing.RegTarget`

> [!NOTE]
>
> **predict\_one**
>
> Predict the output of features `x`.
>
> **Parameters**
>
> - **x**
>
> **Returns**
>
> The prediction.
