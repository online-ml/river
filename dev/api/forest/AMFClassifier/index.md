# AMFClassifier

Aggregated Mondrian Forest classifier for online learning.

This implementation is truly online[^/dev/api/forest/AMFClassifier/#1], in the sense that a single pass is performed, and that predictions can be produced anytime.

Each node in a tree predicts according to the distribution of the labels it contains. This distribution is regularized using a "Jeffreys" prior with parameter `dirichlet`. For each class with `count` labels in the node and `n_samples` samples in it, the prediction of a node is given by

\\(\\frac{count + dirichlet}{n\_{samples} + dirichlet \\times n\_{classes}}\\).

The prediction for a sample is computed as the aggregated predictions of all the subtrees along the path leading to the leaf node containing the sample. The aggregation weights are exponential weights with learning rate `step` and log-loss when `use_aggregation` is `True`.

This computation is performed exactly thanks to a context tree weighting algorithm. More details can be found in the paper cited in the references below.

The final predictions are the average class probabilities predicted by each of the `n_estimators` trees in the forest.

By default, this implementation assumes that all feature values are scaled between 0 and 1. If you cannot assume the minimum and maximum values for each feature, you can use [preprocessing.MinMaxScaler](<https://riverml.xyz/dev/api/preprocessing/MinMaxScaler>) as an initial preprocessing step. This is important because Mondrian trees are highly sensitive to feature scaling, as the distance between a sample and the node's bounding box is calculated as the sum of the distances across all features.

## Parameters

- **n\_estimators**

  *Type* → `int`

  *Default* → `10`

  The number of trees in the forest.
- **step**

  *Type* → `float`

  *Default* → `1.0`

  Step-size for the aggregation weights. Default is 1 for classification with the log-loss, which is usually the best choice.
- **use\_aggregation**

  *Type* → `bool`

  *Default* → `True`

  Controls if aggregation is used in the trees. It is highly recommended to leave it as `True`.
- **dirichlet**

  *Type* → `float`

  *Default* → `0.5`

  Regularization level of the class frequencies used for predictions in each node. A rule of thumb is to set this to `1 / n_classes`, where `n_classes` is the expected number of classes which might appear. Default is `dirichlet = 0.5`, which works well for binary classification problems.
- **split\_pure**

  *Type* → `bool`

  *Default* → `False`

  Controls if nodes that contains only sample of the same class should be split ("pure" nodes). Default is `False`, namely pure nodes are not split, but `True` can be sometimes better.
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

dataset = datasets.Bananas()

model = (
    preprocessing.MinMaxScaler() |
    forest.AMFClassifier(
        n_estimators=10,
        use_aggregation=True,
        dirichlet=0.5,
        seed=1
    )
)

metric = metrics.Accuracy()

evaluate.progressive_val_score(dataset, model, metric)
```

```text
Accuracy: 84.05%
```

## Methods

> [!NOTE]
>
> **learn\_one**
>
> Update the model with a set of features `x` and a label `y`.
>
> **Parameters**
>
> - **x**
> - **y**

> [!NOTE]
>
> **predict\_one**
>
> Predict the label of a set of features `x`.
>
> **Parameters**
>
> - **x** — `dict[base.typing.FeatureName, Any]`
> - **kwargs** — `Any`
>
> **Returns**
>
> `base.typing.ClfTarget | None`: The predicted label.

> [!NOTE]
>
> **predict\_proba\_one**
>
> Predict the probability of each label for a dictionary of features `x`.
>
> **Parameters**
>
> - **x**
>
> **Returns**
>
> A dictionary that associates a probability which each label.

## Notes

Only log\_loss used for the computation of the aggregation weights is supported for now, namely the log-loss for multi-class classification.

[^/dev/api/forest/AMFClassifier/#1]: Mourtada, J., Gaïffas, S., &amp; Scornet, E. (2021). AMF: Aggregated Mondrian forests for online learning. Journal of the Royal Statistical Society Series B: Statistical Methodology, 83(3), 505-533.
