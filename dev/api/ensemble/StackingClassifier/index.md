# StackingClassifier

Stacking for binary classification.

## Parameters

- **models**

  *Type* → `list[base.Classifier]`
- **meta\_classifier**

  *Type* → `base.Classifier`
- **include\_features**

  *Default* → `True`

  Indicates whether or not the original features should be provided to the meta-model along with the predictions from each model.

## Attributes

- **models**

## Examples

```python
from river import compose
from river import datasets
from river import ensemble
from river import evaluate
from river import linear_model as lm
from river import metrics
from river import preprocessing as pp

dataset = datasets.Phishing()

model = compose.Pipeline(
    ('scale', pp.StandardScaler()),
    ('stack', ensemble.StackingClassifier(
        [
            lm.LogisticRegression(),
            lm.PAClassifier(mode=1, C=0.01),
            lm.PAClassifier(mode=2, C=0.01),
        ],
        meta_classifier=lm.LogisticRegression()
    ))
)

metric = metrics.F1()

evaluate.progressive_val_score(dataset, model, metric)
```

```text
F1: 88.14%
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

## References

- [A Kaggler's Guide to Model Stacking in Practice](<http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/>)
