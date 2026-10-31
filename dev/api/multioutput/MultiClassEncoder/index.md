# MultiClassEncoder

Convert a multi-label task into multiclass.

Assigns a class to each unique combination of labels, and proceeds with training the supplied multi-class classifier.

The transformation is done by converting the label set, which could be seen as a binary number, into an integer representing a class. At prediction time, the predicted integer is converted back to a binary number which is the predicted label set.

## Parameters

- **model**

  *Type* → `base.Classifier`

  The classifier used for learning.

## Examples

```python
from river import forest
from river import metrics
from river import multioutput
from river.datasets import synth

dataset = synth.Logical(seed=42, n_tiles=100)

model = multioutput.MultiClassEncoder(
    model=forest.ARFClassifier(seed=7)
)

metric = metrics.multioutput.MicroAverage(metrics.Jaccard())

for x, y in dataset:
   y_pred = model.predict_one(x)
   y_pred = {k: y_pred.get(k, 0) for k in y}
   metric.update(y, y_pred)
   model.learn_one(x, y)

metric
```

```text
MicroAverage(Jaccard): 95.10%
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
> - **x** — `dict[FeatureName, typing.Any]`
> - **y** — `dict[FeatureName, bool]`

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
> - **x** — `dict[FeatureName, typing.Any]`
> - **kwargs** — `typing.Any`
>
> **Returns**
>
> `dict&#91;FeatureName, dict&#91;bool, float&#93;&#93;`: A dictionary that associates a probability which each label.
