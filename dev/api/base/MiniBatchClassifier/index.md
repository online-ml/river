# MiniBatchClassifier

A classifier that can operate on mini-batches.

## Methods

> [!NOTE]
>
> **learn\_many**
>
> Update the model with a mini-batch of features `X` and boolean targets `y`.
>
> **Parameters**
>
> - **X** — `pd.DataFrame`
> - **y** — `pd.Series`

> [!NOTE]
>
> **learn\_one**
>
> Update the model with a set of features `x` and a label `y`.
>
> **Parameters**
>
> - **x** — `dict[base.typing.FeatureName, Any]`
> - **y** — `base.typing.ClfTarget`

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
> `pd.Series`: The predicted labels.

> [!NOTE]
>
> **predict\_one**
>
> Predict the label of a set of features `x`.
>
> **Parameters**
>
> - **x** — `dict[base.typing.FeatureName, Any]`
> - **kwargs** — `Any`
>
> **Returns**
>
> `base.typing.ClfTarget | None`: The predicted label.

> [!NOTE]
>
> **predict\_proba\_many**
>
> Predict the outcome probabilities for each given sample.
>
> **Parameters**
>
> - **X** — `pd.DataFrame`
>
> **Returns**
>
> `pd.DataFrame`: A dataframe with probabilities of `True` and `False` for each sample.

> [!NOTE]
>
> **predict\_proba\_one**
>
> Predict the probability of each label for a dictionary of features `x`.
>
> **Parameters**
>
> - **x** — `dict[base.typing.FeatureName, Any]`
> - **kwargs** — `Any`
>
> **Returns**
>
> `dict&#91;base.typing.ClfTarget, float&#93;`: A dictionary that associates a probability which each label.
