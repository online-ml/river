# SparseRandomProjector

Sparse random projector.

This transformer reduces the dimensionality of inputs by projecting them onto a sparse random projection matrix.

Ping Li et al. recommend using a minimum density of `1 / sqrt(n_features)`. The transformer is not aware of how many features will be seen, so the user must specify the density manually.

## Parameters

- **n\_components**

  *Default* → `10`

  Number of components to project the data onto.
- **density**

  *Default* → `0.1`

  Density of the random projection matrix. The density is defined as the ratio of non-zero components in the matrix. It is equal to `1 - sparsity`.
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
model = preprocessing.SparseRandomProjector(
    n_components=3,
    seed=42
)

for x, y in dataset:
    x = model.transform_one(x)
    print(x)
    break
```

```text
{0: 92.89572746525327, 1: 1344540.5692342375, 2: 0}
```

```python
model = (
    preprocessing.SparseRandomProjector(
        n_components=5,
        seed=42
    ) |
    preprocessing.StandardScaler() |
    linear_model.LinearRegression()
)
evaluate.progressive_val_score(dataset, model, metrics.MAE())
```

```text
MAE: 1.292572
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

- D. Achlioptas. 2003. Database-friendly random projections: Johnson-Lindenstrauss with binary coins. Journal of Computer and System Sciences 66 (2003) 671-687
- Ping Li, Trevor J. Hastie, and Kenneth W. Church. 2006. Very sparse random projections. In Proceedings of the 12th ACM SIGKDD international conference on Knowledge discovery and data mining (KDD'06). ACM, New York, NY, USA, 287-296.
