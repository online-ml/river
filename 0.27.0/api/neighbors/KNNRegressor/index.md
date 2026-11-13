# KNNRegressor

K-Nearest Neighbors regressor.

Samples are stored using a first-in, first-out strategy. The strategy to perform search queries in the data buffer is defined by the `engine` parameter. Predictions are obtained by aggregating the values of the closest n\_neighbors stored samples with respect to a query sample.

## Parameters

- **n\_neighbors**

  *Type* → `int`

  *Default* → `5`

  The number of nearest neighbors to search for.
- **engine**

  *Type* → `BaseNN | None`

  *Default* → `None`

  The search engine used to store the instances and perform search queries. Depending on the choose engine, search will be exact or approximate. Please, consult the documentation of each available search engine for more details on its usage. By default, use the `SWINN` search engine for approximate search queries.
- **aggregation\_method**

  *Type* → `str`

  *Default* → `mean`

  The method to aggregate the target values of neighbors. | 'mean' | 'median' | 'weighted\_mean'

## Examples

```python
from river import datasets
from river import evaluate
from river import metrics
from river import neighbors
from river import preprocessing

dataset = datasets.TrumpApproval()

model = neighbors.KNNRegressor()
evaluate.progressive_val_score(dataset, model, metrics.RMSE())
```

```text
RMSE: 1.427743
```

## Methods

> [!NOTE]
>
> **learn\_one**
>
> Fits to a set of features `x` and a real-valued target `y`.
>
> **Parameters**
>
> - **x** — `dict[base.typing.FeatureName, typing.Any]`
> - **y** — `base.typing.RegTarget`

> [!NOTE]
>
> **predict\_one**
>
> Predict the output of features `x`.
>
> **Parameters**
>
> - **x** — `dict[base.typing.FeatureName, typing.Any]`
> - **kwargs** — `typing.Any`
>
> **Returns**
>
> `base.typing.RegTarget`: The prediction.
