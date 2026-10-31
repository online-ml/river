# AMFRegressor

Aggregated Mondrian Forest regressor for online learning.

This algorithm is truly online, in the sense that a single pass is performed, and that predictions can be produced anytime.

Each node in a tree predicts according to the average of the labels it contains. The prediction for a sample is computed as the aggregated predictions of all the subtrees along the path leading to the leaf node containing the sample. The aggregation weights are exponential weights with learning rate `step` using a squared loss when `use_aggregation` is `True`.

This computation is performed exactly thanks to a context tree weighting algorithm. More details can be found in the original paper[^/dev/api/forest/AMFRegressor/#1].

The final predictions are the average of the predictions of each of the `n_estimators` trees in the forest.

By default, this implementation assumes that all feature values are scaled between 0 and 1. If you cannot assume the minimum and maximum values for each feature, you can use [preprocessing.MinMaxScaler](<https://riverml.xyz/dev/api/preprocessing/MinMaxScaler>) as an initial preprocessing step. This is important because Mondrian trees are highly sensitive to feature scaling, as the distance between a sample and the node's bounding box is calculated as the sum of the distances across all features.

## Parameters

- **n\_estimators**

  *Type* → `int`

  *Default* → `10`

  The number of trees in the forest.
- **step**

  *Type* → `float`

  *Default* → `1.0`

  Step-size for the aggregation weights.
- **use\_aggregation**

  *Type* → `bool`

  *Default* → `True`

  Controls if aggregation is used in the trees. It is highly recommended to leave it as `True`.
- **max\_nodes**

  *Type* → `int | None`

  *Default* → `None`

  Maximum number of nodes allowed per tree. No new splits will occur in a tree once this limit is reached. If `None`, trees grow without bound. Setting this limits memory usage at the cost of potentially less accurate predictions.
- **seed**

  *Type* → `int | None`

  *Default* → `None`

  Random seed for reproducibility.

## Attributes

- **models**

## Examples

```python
from river import datasets
from river import evaluate
from river import forest
from river import metrics
from river import preprocessing

dataset = datasets.TrumpApproval()

model = (
    preprocessing.MinMaxScaler() |
    forest.AMFRegressor(seed=42)
)

metric = metrics.MAE()
evaluate.progressive_val_score(dataset, model, metric)
```

```text
MAE: 0.427341
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
> - **x**
> - **y**

> [!NOTE]
>
> **predict\_one**
>
> Predict the output of features `x`.
>
> **Parameters**
>
> - **x**
>
> **Returns**
>
> The prediction.

[^/dev/api/forest/AMFRegressor/#1]: Mourtada, J., Gaïffas, S., &amp; Scornet, E. (2021). AMF: Aggregated Mondrian forests for online learning. Journal of the Royal Statistical Society Series B: Statistical Methodology, 83(3), 505-533.
