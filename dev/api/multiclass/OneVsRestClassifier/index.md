# OneVsRestClassifier

One-vs-the-rest (OvR) multiclass strategy.

This strategy consists in fitting one binary classifier per class. Because we are in a streaming context, the number of classes isn't known from the start. Hence, new classifiers are instantiated on the fly. Likewise, the predicted probabilities will only include the classes seen up to a given point in time.

Note that this classifier supports mini-batches as well as single instances.

The computational complexity for both learning and predicting grows linearly with the number of classes. If you have a very large number of classes, then you might want to consider using an [`multiclass.OutputCodeClassifier`](<https://riverml.xyz/dev/api/multiclass/OutputCodeClassifier>) instead.

## Parameters

- **classifier**

  *Type* → `base.Classifier`

  A binary classifier, although a multi-class classifier will work too.

## Attributes

- **classifiers** (`dict`)

  A mapping between classes and classifiers.

## Examples

```python
import pandas as pd
from river import datasets
from river import evaluate
from river import linear_model
from river import metrics
from river import multiclass
from river import preprocessing

dataset = datasets.ImageSegments()

scaler = preprocessing.StandardScaler()
ovr = multiclass.OneVsRestClassifier(linear_model.LogisticRegression())
model = scaler | ovr

metric = metrics.MacroF1()

evaluate.progressive_val_score(dataset, model, metric)
```

```text
MacroF1: 77.46%
```

This estimator also supports mini-batching. The whole pipeline — the `StandardScaler` and the one-vs-rest `LogisticRegression`s — is trained and queried on mini-batches of a dataframe:

```python
model = preprocessing.StandardScaler() | multiclass.OneVsRestClassifier(
    linear_model.LogisticRegression()
)

metric = metrics.MacroF1()

for X in pd.read_csv(dataset.path, chunksize=64):
    y = X.pop('category')
    y_pred = model.predict_many(X)
    model.learn_many(X, y)
    for y_true, y_p in zip(y, y_pred):
        if y_p is not None:
            metric.update(y_true, y_p)

metric
```

```text
MacroF1: 59.77%
```

The mini-batch methods are dataframe-agnostic: any narwhals-supported eager backend (pandas, polars, pyarrow, ...) can be passed to `learn_many`/`predict_many`, and the predictions are returned in that same backend.

## Methods

> [!NOTE]
>
> **learn\_many**

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
> - **kwargs**

> [!NOTE]
>
> **predict\_many**

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
> **predict\_proba\_many**

> [!NOTE]
>
> **predict\_proba\_one**
>
> Predict the probability of each label for a dictionary of features `x`.
>
> **Parameters**
>
> - **x**
> - **kwargs**
>
> **Returns**
>
> A dictionary that associates a probability which each label.
