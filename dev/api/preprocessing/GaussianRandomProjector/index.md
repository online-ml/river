# GaussianRandomProjector

Gaussian random projector.

This transformer reduces the dimensionality of inputs through Gaussian random projection.

The components of the random projections matrix are drawn from `N(0, 1 / n_components)`.

## Parameters

- **n\_components**

  *Default* → `10`

  Number of components to project the data onto.
- **seed**

  *Type* → `int | None`

  *Default* → `None`

  Random seed for reproducibility.

## Examples

```python
from river import datasets
from river import evaluate
from river import linear_model
from river import metrics
from river import preprocessing

dataset = datasets.TrumpApproval()
model = preprocessing.GaussianRandomProjector(
    n_components=3,
    seed=42
)

for x, y in dataset:
    x = model.transform_one(x)
    print(x)
    break
```

```text
{0: -61289.371..., 1: 141312.510..., 2: 279165.993...}
```

```python
model = (
    preprocessing.GaussianRandomProjector(
        n_components=5,
        seed=42
    ) |
    preprocessing.StandardScaler() |
    linear_model.LinearRegression()
)
evaluate.progressive_val_score(dataset, model, metrics.MAE())
```

```text
MAE: 0.933...
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

- [Gaussian random projection](<https://en.wikipedia.org/wiki/Random_projection#Gaussian_random_projection>)
- [scikit-learn random projections module](<https://scikit-learn.org/stable/modules/random_projection.html>)
