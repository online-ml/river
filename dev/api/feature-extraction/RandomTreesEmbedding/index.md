# RandomTreesEmbedding

Embed samples according to the leaves of random trees.

This transformer builds an ensemble of totally random trees and encodes each sample with the leaf reached in every tree. The output is sparse: exactly one binary feature is active per tree.

This is the online counterpart of feeding a linear model with random-tree leaf indicators. By default, this implementation assumes that each feature has values comprised between 0 and 1. If this is not the case, then you can manually specify the limits via the `limits` argument. If you do not know the limits in advance, then you can use a [`preprocessing.MinMaxScaler`](<https://riverml.xyz/dev/api/preprocessing/MinMaxScaler>) as an initial preprocessing step.

The trees are built lazily from the first sample that is observed, either via `transform_one` or `learn_one`. If new features appear later in the stream, then the forest is rebuilt so that future splits may use them. Unless specified in `limits`, newly observed features are also assumed to lie in the range `\[0, 1\]`.

## Parameters

- **n\_trees**

  *Default* → `10`

  Number of trees in the ensemble.
- **height**

  *Default* → `8`

  Height of each tree. A tree of height `h` contains `2 ** h` leaves.
- **limits**

  *Type* → `dict[base.typing.FeatureName, tuple[float, float]] | None`

  *Default* → `None`

  Specifies the range of each feature. By default each feature is assumed to be in range `\[0, 1\]`. If the feature ranges are unknown beforehand, then use a `preprocessing.MinMaxScaler` upstream.
- **seed**

  *Type* → `int | None`

  *Default* → `None`

  Random seed for reproducibility.

## Examples

```python
from river import feature_extraction as fx
from river import linear_model as lm
from river import optim
from river import preprocessing

embedding = fx.RandomTreesEmbedding(n_trees=3, height=2, seed=42)
len(embedding.transform_one({'x': 0.3, 'y': 0.7}))
```

```text
3
```

```python
model = (
    preprocessing.MinMaxScaler() |
    fx.RandomTreesEmbedding(n_trees=5, height=3, seed=42) |
    lm.LogisticRegression(optimizer=optim.SGD(0.1))
)
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

- [Geurts, P., Ernst, D., and Wehenkel, L. (2006). Extremely randomized trees. Machine Learning, 63(1), 3-42.](<https://link.springer.com/article/10.1007/s10994-006-6226-1>)
- [scikit-learn random trees embedding](<https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomTreesEmbedding.html>)
