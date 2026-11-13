# VarianceThreshold

Removes low-variance features.

## Parameters

- **threshold**

  *Type* → `float`

  *Default* → `0`

  Only features with a variance above the threshold will be kept.
- **min\_samples**

  *Type* → `int`

  *Default* → `2`

  The minimum number of samples required to perform selection.

## Attributes

- **variances** (`dict`)

  The variance of each feature.

## Examples

```python
from river import feature_selection
from river import stream

X = [
    [0, 2, 0, 3],
    [0, 1, 4, 3],
    [0, 1, 1, 3]
]

selector = feature_selection.VarianceThreshold()

for x, _ in stream.iter_array(X):
    selector.learn_one(x)
    print(selector.transform_one(x))
```

```text
{0: 0, 1: 2, 2: 0, 3: 3}
{1: 1, 2: 4}
{1: 1, 2: 1}
```

## Methods

> [!NOTE]
>
> **check\_feature**

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
> - **x** — `dict[base.typing.FeatureName, typing.Any]`

> [!NOTE]
>
> **transform\_one**
>
> Transform a set of features `x`.
>
> **Parameters**
>
> - **x** — `dict[base.typing.FeatureName, typing.Any]`
>
> **Returns**
>
> `dict&#91;base.typing.FeatureName, typing.Any&#93;`: The transformed values.
