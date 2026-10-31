# VotingClassifier

Voting classifier.

A classification is made by aggregating the predictions of each model in the ensemble. The probabilities for each class are summed up if `use_probabilities` is set to `True`. If not, the probabilities are ignored and each prediction is weighted the same. In this case, it's important that you use an odd number of classifiers. A random class will be picked if the number of classifiers is even.

## Parameters

- **models**

  *Type* → `list[base.Classifier]`

  The classifiers.
- **use\_probabilities**

  *Default* → `True`

  Whether or to weight each prediction with its associated probability.

## Attributes

- **models**

## Examples

```python
from river import datasets
from river import ensemble
from river import evaluate
from river import linear_model
from river import metrics
from river import naive_bayes
from river import preprocessing
from river import tree

dataset = datasets.Phishing()

model = (
    preprocessing.StandardScaler() |
    ensemble.VotingClassifier([
        linear_model.LogisticRegression(),
        tree.HoeffdingTreeClassifier(),
        naive_bayes.GaussianNB()
    ])
)

metric = metrics.F1()

evaluate.progressive_val_score(dataset, model, metric)
```

```text
F1: 87.04%
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

> [!NOTE]
>
> **predict\_one**
>
> Predict the label of a set of features `x`.
>
> **Parameters**
>
> - **x** — `dict[base.typing.FeatureName, Any]`
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
