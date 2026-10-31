# FeatureHasher

Implements the hashing trick.

Each pair of (name, value) features is hashed into a random integer in `[0, n_features)`, using the signed 32-bit MurmurHash3 of the feature's token. String values are hashed as `"name=value"` tokens and contribute `1`; numeric values are hashed under `"name"` and contribute the value itself.

The hashing is performed in Rust, so the whole transform of an example happens in a single native call.

## Parameters

- **n\_features**

  *Default* → `1048576`

  The number by which each hash will be moduloed by.
- **seed**

  *Type* → `int | None`

  *Default* → `None`

  Set the seed to produce identical results. When `None`, a random seed is drawn, so two instances will hash features to different buckets.
- **alternate\_sign**

  *Type* → `bool`

  *Default* → `True`

  When `True` (the default), the sign bit of the hash is used to negate half of the contributions. This keeps the expected value of each bucket at zero, so hash collisions between unrelated features tend to cancel out rather than accumulate, which is especially helpful for small `n_features`. This matches scikit-learn's `FeatureHasher`.

## Examples

```python
import river

hasher = river.preprocessing.FeatureHasher(n_features=10, seed=42)

X = [
    {'dog': 1, 'cat': 2, 'elephant': 4},
    {'dog': 2, 'run': 5}
]
for x in X:
    print(hasher.transform_one(x))
```

```text
{5: -3, 7: 2}
{5: 2, 9: -5}
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

## References

- [Wikipedia article on feature vectorization using the hashing trick](<https://en.wikipedia.org/wiki/Feature_hashing#Algorithms>)
- [Weinberger et al. (2009), Feature Hashing for Large Scale Multitask Learning](<https://arxiv.org/abs/0902.2206>)
