# BanditClassifier

Bandit-based model selection for classification.

Each model is associated with an arm. At each `learn_one` call, the policy decides which arm/model to pull. The reward is the performance of the model on the provided sample. The `predict_one` and `predict_proba_one` methods use the current best model.

## Parameters

- **models**

  The models to select from.
- **metric**

  *Type* → `metrics.base.ClassificationMetric`

  The metric that is used to measure the performance of each model.
- **policy**

  *Type* → `bandit.base.Policy`

  The bandit policy to use.

## Attributes

- **best\_model**
- **models**

## Examples

```python
from river import bandit
from river import datasets
from river import evaluate
from river import linear_model
from river import metrics
from river import model_selection
from river import optim
from river import preprocessing

models = [
    linear_model.LogisticRegression(optimizer=optim.SGD(lr=lr))
    for lr in [0.0001, 0.001, 1e-05, 0.01]
]

dataset = datasets.Phishing()
model = (
    preprocessing.StandardScaler() |
    model_selection.BanditClassifier(
        models,
        metric=metrics.Accuracy(),
        policy=bandit.EpsilonGreedy(
            epsilon=0.1,
            decay=0.001,
            burn_in=20,
            seed=42
        )
    )
)
metric = metrics.Accuracy()

evaluate.progressive_val_score(dataset, model, metric)
```

```text
Accuracy: 88.96%
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
> - **x**
>
> **Returns**
>
> A dictionary that associates a probability which each label.
