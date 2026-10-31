# ADWINBaggingClassifier

ADWIN Bagging classifier.

ADWIN Bagging [^/dev/api/ensemble/ADWINBaggingClassifier/#1] is the online bagging method of Oza and Russell [^/dev/api/ensemble/ADWINBaggingClassifier/#2] with the addition of the `ADWIN` algorithm as a change detector. If concept drift is detected, the worst member of the ensemble (based on the error estimation by ADWIN) is replaced by a new (empty) classifier.

## Parameters

- **model**

  *Type* → `base.Classifier`

  The classifier to bag.
- **n\_models**

  *Default* → `10`

  The number of models in the ensemble.
- **seed**

  *Type* → `int | None`

  *Default* → `None`

  Random number generator seed for reproducibility.

## Attributes

- **models**

## Examples

```python
from river import datasets
from river import ensemble
from river import evaluate
from river import linear_model
from river import metrics
from river import optim
from river import preprocessing

dataset = datasets.Phishing()

model = ensemble.ADWINBaggingClassifier(
    model=(
        preprocessing.StandardScaler() |
        linear_model.LogisticRegression()
    ),
    n_models=3,
    seed=42
)

metric = metrics.F1()

evaluate.progressive_val_score(dataset, model, metric)
```

```text
F1: 87.65%
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
> - **kwargs**

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
> Averages the predictions of each classifier.
>
> **Parameters**
>
> - **x**
> - **kwargs**

[^/dev/api/ensemble/ADWINBaggingClassifier/#1]: Albert Bifet, Geoff Holmes, Bernhard Pfahringer, Richard Kirkby, and Ricard Gavaldà. "New ensemble methods for evolving data streams." In 15th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2009.

[^/dev/api/ensemble/ADWINBaggingClassifier/#2]: Oza, N., Russell, S. "Online bagging and boosting." In: Artificial Intelligence and Statistics 2001, pp. 105–112. Morgan Kaufmann, 2001.
