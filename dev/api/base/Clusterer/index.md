# Clusterer

A clustering model.

## Methods

> [!NOTE]
>
> **learn\_one**
>
> Update the model with a set of features `x`.
>
> **Parameters**
>
> - **x** — `dict[typing.FeatureName, Any]`

> [!NOTE]
>
> **predict\_one**
>
> Predicts the cluster number for a set of features `x`.
>
> **Parameters**
>
> - **x** — `dict[typing.FeatureName, Any]`
>
> **Returns**
>
> `int`: A cluster number.
