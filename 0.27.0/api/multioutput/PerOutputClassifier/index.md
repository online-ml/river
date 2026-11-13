# PerOutputClassifier

A multi-output model that trains one independent classifier per output.

This model does not use the prediction of one output as a feature for the next. Each output is modelled by its own copy of the base classifier, trained independently. (This is the streaming equivalent of scikit-learn's `MultiOutputClassifier`).

The set of outputs isn't known from the start in a streaming setting, new classifiers are instantiated on the fly, one per output key encountered in `y`.

## Parameters

- **classifier**

  *Type* → `base.Classifier`

  A classifier model used for each label.

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
model |= multioutput.PerOutputClassifier(
    classifier=linear_model.LogisticRegression(),
)

metric = metrics.multioutput.MicroAverage(metrics.Jaccard())

for x, y in dataset:
    y_pred = model.predict_one(x)
    y_pred = {k: y_pred.get(k, False) for k in y}
    metric.update(y, y_pred)
    model.learn_one(x, y)

metric
```

```text
MicroAverage(Jaccard): 41.82%
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
> - **x**
> - **kwargs**
>
> **Returns**
>
> The predicted labels.

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
