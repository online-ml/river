# SKL2RiverClassifier

Compatibility layer from scikit-learn to River for classification.

## Parameters

- **estimator**

  *Type* → `sklearn_base.BaseEstimator`

  A scikit-learn regressor which has a `partial_fit` method.
- **classes**

  *Type* → `list[base.typing.ClfTarget]`

## Examples

```python
from river import compat
from river import evaluate
from river import metrics
from river import preprocessing
from river import stream
from sklearn import linear_model
from sklearn import datasets

dataset = stream.iter_sklearn_dataset(
    dataset=datasets.load_breast_cancer(),
    shuffle=True,
    seed=42
)

model = preprocessing.StandardScaler()
model |= compat.convert_sklearn_to_river(
    estimator=linear_model.SGDClassifier(
        loss='log_loss',
        eta0=0.01,
        learning_rate='constant'
    ),
    classes=[False, True]
)

metric = metrics.LogLoss()

evaluate.progressive_val_score(dataset, model, metric)
```

```text
LogLoss: 0.198029
```

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
> - **x** — `dict[base.typing.FeatureName, typing.Any]`
> - **y** — `base.typing.ClfTarget`

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
> - **x** — `dict[base.typing.FeatureName, typing.Any]`
> - **kwargs** — `typing.Any`
>
> **Returns**
>
> `base.typing.ClfTarget`: The predicted label.

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
> - **x** — `dict[base.typing.FeatureName, typing.Any]`
> - **kwargs** — `typing.Any`
>
> **Returns**
>
> `dict&#91;base.typing.ClfTarget, float&#93;`: A dictionary that associates a probability which each label.
