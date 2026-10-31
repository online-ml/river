# MiniBatchTransformer

A transform that can operate on mini-batches.

## Methods

> [!NOTE]
>
> **learn\_many**
>
> Update with a mini-batch of features.
>
> A lot of transformers don't actually have to do anything during the `learn_many` step because they are stateless. For this reason the default behavior of this function is to do nothing. Transformers that however do something during the `learn_many` can override this method.
>
> **Parameters**
>
> - **X** — `pd.DataFrame`

> [!NOTE]
>
> **learn\_one**
>
> Update with a set of features `x`.
>
> A lot of transformers don't actually have to do anything during the `learn_one` step because they are stateless. For this reason the default behavior of this function is to do nothing. Transformers that however do something during the `learn_one` can override this method.
>
> **Parameters**
>
> - **x** — `dict[base.typing.FeatureName, Any]`

> [!NOTE]
>
> **transform\_many**
>
> Transform a mini-batch of features.
>
> **Parameters**
>
> - **X** — `pd.DataFrame`
>
> **Returns**
>
> `pd.DataFrame`: A new DataFrame.

> [!NOTE]
>
> **transform\_one**
>
> Transform a set of features `x`.
>
> **Parameters**
>
> - **x** — `dict[base.typing.FeatureName, Any]`
>
> **Returns**
>
> `dict&#91;base.typing.FeatureName, Any&#93;`: The transformed values.
