# ALMAClassifier

Approximate Large Margin Algorithm (ALMA).

## Parameters

- **p**

  *Type* → `float`

  *Default* → `2`
- **alpha**

  *Type* → `float`

  *Default* → `0.9`
- **B**

  *Type* → `float`

  *Default* → `1.1111111111111112`
- **C**

  *Type* → `float`

  *Default* → `1.4142135623730951`

## Attributes

- **w** (`collections.defaultdict`)

  The current weights.
- **k** (`int`)

  The number of instances seen during training.

## Examples

```python
from river import datasets
from river import evaluate
from river import linear_model
from river import metrics
from river import preprocessing

dataset = datasets.Phishing()

model = (
    preprocessing.StandardScaler() |
    linear_model.ALMAClassifier()
)

metric = metrics.Accuracy()

evaluate.progressive_val_score(dataset, model, metric)
```

```text
Accuracy: 82.56%
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
> - **x** — `dict[base.typing.FeatureName, float]`
> - **kwargs** — `typing.Any`
>
> **Returns**
>
> `dict&#91;base.typing.ClfTarget, float&#93;`: A dictionary that associates a probability which each label.

## References

- [Gentile, Claudio. "A new approximate maximal margin classification algorithm." Journal of Machine Learning Research 2.Dec (2001): 213-242](<http://www.jmlr.org/papers/volume2/gentile01a/gentile01a.pdf>)
