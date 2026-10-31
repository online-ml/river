# MultiTargetRegressor

Multi-target regressor.

## Methods

> [!NOTE]
>
> **learn\_one**
>
> Fits to a set of features `x` and a real-valued target `y`.
>
> **Parameters**
>
> - **x** — `dict[FeatureName, typing.Any]`
> - **y** — `dict[FeatureName, RegTarget]`
> - **kwargs** — `typing.Any`

> [!NOTE]
>
> **predict\_one**
>
> Predict the outputs of features `x`.
>
> **Parameters**
>
> - **x** — `dict[FeatureName, typing.Any]`
>
> **Returns**
>
> `dict&#91;FeatureName, RegTarget&#93;`: The predictions.
