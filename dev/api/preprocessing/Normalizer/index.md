# Normalizer

Scales a set of features so that it has unit norm.

This is particularly useful when used after a [`feature_extraction.TFIDF`](<https://riverml.xyz/dev/api/feature-extraction/TFIDF>).

## Parameters

- **order**

  *Default* → `2`

  Order of the norm (e.g. 2 corresponds to the \\(L^2\\) norm).

## Examples

```python
from river import preprocessing
from river import stream

scaler = preprocessing.Normalizer(order=2)

X = [[4, 1, 2, 2],
     [1, 3, 9, 3],
     [5, 7, 5, 1]]

for x, _ in stream.iter_array(X):
    print(scaler.transform_one(x))
```

```text
{0: 0.8, 1: 0.2, 2: 0.4, 3: 0.4}
{0: 0.1, 1: 0.3, 2: 0.9, 3: 0.3}
{0: 0.5, 1: 0.7, 2: 0.5, 3: 0.1}
```

## Methods

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
