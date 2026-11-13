# OneVsOneClassifier

One-vs-One (OvO) multiclass strategy.

This strategy consists in fitting one binary classifier for each pair of classes. Because we are in a streaming context, the number of classes isn't known from the start, hence new classifiers are instantiated on the fly.

The number of classifiers is `k * (k - 1) / 2`, where `k` is the number of classes. However, each call to `learn_one` only requires training `k - 1` models. Indeed, only the models that pertain to the given label have to be trained. Meanwhile, making a prediction requires going through each and every model.

## Parameters

- **classifier**

  *Type* → `base.Classifier`

  A binary classifier, although a multi-class classifier will work too.

## Attributes

- **classifiers** (`dict`)

  A mapping between pairs of classes and classifiers. The keys are tuples which contain a pair of classes. Each pair is sorted in lexicographical order.

## Examples

```python
from river import datasets
from river import evaluate
from river import linear_model
from river import metrics
from river import multiclass
from river import preprocessing

dataset = datasets.ImageSegments()

scaler = preprocessing.StandardScaler()
ovo = multiclass.OneVsOneClassifier(linear_model.LogisticRegression())
model = scaler | ovo

metric = metrics.MacroF1()

evaluate.progressive_val_score(dataset, model, metric)
```

```text
MacroF1: 80.76%
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
> - **kwargs** — `typing.Any`

> [!NOTE]
>
> **predict\_one**
>
> Predict the label of a set of features `x`.
>
> **Parameters**
>
> - **x** — `dict[base.typing.FeatureName, typing.Any]`
> - **kwargs** — `typing.Any`
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
> - **x** — `dict[base.typing.FeatureName, Any]`
> - **kwargs** — `Any`
>
> **Returns**
>
> `dict&#91;base.typing.ClfTarget, float&#93;`: A dictionary that associates a probability which each label.
