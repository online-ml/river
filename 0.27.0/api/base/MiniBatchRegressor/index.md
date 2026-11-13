# MiniBatchRegressor

A regressor that can operate on mini-batches.

## Methods

> [!NOTE]
>
> **learn\_many**
>
> Update the model with a mini-batch of features `X` and real-valued targets `y`.
>
> **Parameters**
>
> - **X** — `pd.DataFrame`
> - **y** — `pd.Series`

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
> **predict\_many**
>
> Predict the outcome for each given sample.
>
> **Parameters**
>
> - **X** — `pd.DataFrame`
>
> **Returns**
>
> `pd.Series`: The predicted outcomes.

> [!NOTE]
>
> **predict\_one**
>
> Predict the output of features `x`.
>
> **Parameters**
>
> - **x** — `dict[base.typing.FeatureName, Any]`
>
> **Returns**
>
> `base.typing.RegTarget`: The prediction.
