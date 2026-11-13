# KNNClassifier

K-Nearest Neighbors (KNN) for classification.

Samples are stored using a first-in, first-out strategy. The strategy to perform search queries in the data buffer is defined by the `engine` parameter.

## Parameters

- **n\_neighbors**

  *Type* → `int`

  *Default* → `5`

  The number of nearest neighbors to search for.
- **engine**

  *Type* → `BaseNN | None`

  *Default* → `None`

  The search engine used to store the instances and perform search queries. Depending on the choose engine, search will be exact or approximate. Please, consult the documentation of each available search engine for more details on its usage. By default, use the `SWINN` search engine for approximate search queries.
- **weighted**

  *Type* → `bool`

  *Default* → `True`

  Weight the contribution of each neighbor by its inverse distance.
- **cleanup\_every**

  *Type* → `int`

  *Default* → `0`

  This determines at which rate old classes are cleaned up. Classes that have been seen in the past but that are not present in the current window are dropped. Classes are never dropped when this is set to 0.
- **softmax**

  *Type* → `bool`

  *Default* → `False`

  Whether or not to use softmax normalization to normalize the neighbors contributions. Votes are divided by the total number of votes if this is `False`.

## Examples

```python
import functools
from river import datasets
from river import evaluate
from river import metrics
from river import neighbors
from river import preprocessing
from river import utils

dataset = datasets.Phishing()
```

To select a custom distance metric which takes one or several parameter, you can wrap your chosen distance using `functools.partial`:

```python
l1_dist = functools.partial(utils.math.minkowski_distance, p=1)

model = (
    preprocessing.StandardScaler() |
    neighbors.KNNClassifier(
        engine=neighbors.SWINN(
            dist_func=l1_dist,
            seed=42
        )
    )
)

evaluate.progressive_val_score(dataset, model, metrics.Accuracy())
```

```text
Accuracy: 89.51%
```

## Methods

> [!NOTE]
>
> **clean\_up\_classes**
>
> Clean up classes added to the window.
>
> Classes that are added (and removed) from the window may no longer be valid. This method cleans up the window and and ensures only known classes are added, and we do not consider "None" a class. It is called every `cleanup_every` step, or can be called manually.

> [!NOTE]
>
> **learn\_one**
>
> Update the model with a set of features `x` and a label `y`.
>
> **Parameters**
>
> - **x** — `dict[base.typing.FeatureName, typing.Any]`
> - **y** — `base.typing.ClfTarget`

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
> - **x** — `dict[base.typing.FeatureName, typing.Any]`
> - **kwargs** — `typing.Any`
>
> **Returns**
>
> `dict&#91;base.typing.ClfTarget, float&#93;`: A dictionary that associates a probability which each label.

## Notes

Note that since the window is moving and we keep track of all classes that are added at some point, a class might be returned in a result (with a value of 0) if it is no longer in the window. You can call model.clean\_up\_classes(), or set `cleanup_every` to a non-zero value.
