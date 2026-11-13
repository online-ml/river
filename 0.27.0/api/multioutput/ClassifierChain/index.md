# ClassifierChain

A multi-output model that arranges classifiers into a chain.

This will create one model per output. The prediction of the first output will be used as a feature in the second model. The prediction for the second output will be used as a feature for the third model, etc. This "chain model" is therefore capable of capturing dependencies between outputs.

## Parameters

- **model**

  *Type* → `base.Classifier`

  A classifier model used for each label.
- **order**

  *Type* → `list | None`

  *Default* → `None`

  A list with the targets order in which to construct the chain. If `None` then the order will be inferred from the order of the keys in the target.

## Examples

```python
import random

from river import datasets
from river import feature_selection
from river import linear_model
from river import metrics
from river import multioutput
from river import preprocessing

dataset = list(datasets.Yeast())
random.Random(42).shuffle(dataset)

model = feature_selection.VarianceThreshold(threshold=0.01)
model |= preprocessing.StandardScaler()
model |= multioutput.ClassifierChain(
    model=linear_model.LogisticRegression(),
    order=[f"Class{i}" for i in range(1, 15)],
)

metric = metrics.multioutput.MicroAverage(metrics.Jaccard())

for x, y in dataset:
    y_pred = model.predict_one(x)
    metric.update(y, y_pred)
    model.learn_one(x, y)

metric
```

```text
MicroAverage(Jaccard): 42.31%
```

## Methods

> [!NOTE]
>
> **learn\_one**
>
> Update the model with a set of features `x` and the labels `y`.
>
> **Parameters**
>
> - **x**
> - **y**
> - **kwargs**

> [!NOTE]
>
> **predict\_one**
>
> Predict the labels of a set of features `x`.
>
> **Parameters**
>
> - **x** — `dict[FeatureName, typing.Any]`
> - **kwargs** — `typing.Any`
>
> **Returns**
>
> `dict&#91;FeatureName, bool&#93;`: The predicted labels.

> [!NOTE]
>
> **predict\_proba\_one**
>
> Predict the probability of each label appearing given dictionary of features `x`.
>
> **Parameters**
>
> - **x**
> - **kwargs**
>
> **Returns**
>
> A dictionary that associates a probability which each label.

## References

- [Multi-Output Chain Models and their Application in Data Streams](<https://jmread.github.io/talks/2019_03_08-Imperial_Stats_Seminar.pdf>)
