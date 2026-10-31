# MultiLabelClassifier

Multi-label classifier.

## Methods

> [!NOTE]
>
> **learn\_one**
>
> Update the model with a set of features `x` and the labels `y`.
>
> **Parameters**
>
> - **x** — `dict[FeatureName, typing.Any]`
> - **y** — `dict[FeatureName, bool]`

> [!NOTE]
>
> **predict\_one**
>
> Predict the labels of a set of features `x`.
>
> **Parameters**
>
> - **x** — `dict[FeatureName, typing.Any]`
> - **kwargs** — `typing.Any`
>
> **Returns**
>
> `dict&#91;FeatureName, bool&#93;`: The predicted labels.

> [!NOTE]
>
> **predict\_proba\_one**
>
> Predict the probability of each label appearing given dictionary of features `x`.
>
> **Parameters**
>
> - **x** — `dict[FeatureName, typing.Any]`
> - **kwargs** — `typing.Any`
>
> **Returns**
>
> `dict&#91;FeatureName, dict&#91;bool, float&#93;&#93;`: A dictionary that associates a probability which each label.
