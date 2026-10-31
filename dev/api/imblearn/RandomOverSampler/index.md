# RandomOverSampler

Random over-sampling.

This is a wrapper for classifiers. It will train the provided classifier by over-sampling the stream of given observations so that the class distribution seen by the classifier follows a given desired distribution. The implementation is a discrete version of reverse rejection sampling.

See [Working with imbalanced data](</user-guide/imbalanced-learning>) for example usage.

## Parameters

- **classifier**

  *Type* → `base.Classifier`
- **desired\_dist**

  *Type* → `dict`

  The desired class distribution. The keys are the classes whilst the values are the desired class percentages. The values must sum up to 1.
- **seed**

  *Type* → `int | None`

  *Default* → `None`

  Random seed for reproducibility.

## Examples

```python
from river import datasets
from river import evaluate
from river import imblearn
from river import linear_model
from river import metrics
from river import preprocessing

model = imblearn.RandomOverSampler(
    (
        preprocessing.StandardScaler() |
        linear_model.LogisticRegression()
    ),
    desired_dist={False: 0.4, True: 0.6},
    seed=42
)

dataset = datasets.CreditCard().take(3000)

metric = metrics.LogLoss()

evaluate.progressive_val_score(dataset, model, metric)
```

```text
LogLoss: 0.0457...
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
> - **x** — `dict[base.typing.FeatureName, Any]`
> - **y** — `base.typing.ClfTarget`
> - **kwargs**

> [!NOTE]
>
> **predict\_one**
>
> Predict the label of a set of features `x`.
>
> **Parameters**
>
> - **x**
> - **kwargs**
>
> **Returns**
>
> The predicted label.

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
