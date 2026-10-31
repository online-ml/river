# SGTClassifier

Stochastic Gradient Tree[^/dev/api/tree/SGTClassifier/#1] for binary classification.

Binary decision tree classifier that minimizes the binary cross-entropy to guide its growth.

Stochastic Gradient Trees (SGT) directly minimize a loss function to guide tree growth and update their predictions. Thus, they differ from other incrementally tree learners that do not directly optimize the loss, but data impurity-related heuristics.

## Parameters

- **delta**

  *Type* → `float`

  *Default* → `1e-07`

  Define the significance level of the F-tests performed to decide upon creating splits or updating predictions.
- **grace\_period**

  *Type* → `int`

  *Default* → `200`

  Interval between split attempts or prediction updates.
- **init\_pred**

  *Type* → `float`

  *Default* → `0.0`

  Initial value predicted by the tree.
- **max\_depth**

  *Type* → `int | None`

  *Default* → `None`

  The maximum depth the tree might reach. If set to `None`, the trees will grow indefinitely.
- **lambda\_value**

  *Type* → `float`

  *Default* → `0.1`

  Positive float value used to impose a penalty over the tree's predictions and force them to become smaller. The greater the lambda value, the more constrained are the predictions.
- **gamma**

  *Type* → `float`

  *Default* → `1.0`

  Positive float value used to impose a penalty over the tree's splits and force them to be avoided when possible. The greater the gamma value, the smaller the chance of a split occurring.
- **nominal\_attributes**

  *Type* → `list | None`

  *Default* → `None`

  List with identifiers of the nominal attributes. If None, all features containing numbers are assumed to be numeric.
- **feature\_quantizer**

  *Type* → `tree.splitter.Quantizer | None`

  *Default* → `None`

  The algorithm used to quantize numeric features. Either a static quantizer (as in the original implementation) or a dynamic quantizer can be used. The correct choice and setup of the feature quantizer is a crucial step to determine the performance of SGTs. Feature quantizers are akin to the attribute observers used in Hoeffding Trees. By default, an instance of `tree.splitter.StaticQuantizer` (with default parameters) is used if this parameter is not set.

## Attributes

- **height**
- **n\_branches**
- **n\_leaves**
- **n\_node\_updates**
- **n\_nodes**
- **n\_observations**
- **n\_splits**

## Examples

```python
from river import datasets
from river import evaluate
from river import metrics
from river import tree

dataset = datasets.Phishing()
model = tree.SGTClassifier(
    feature_quantizer=tree.splitter.StaticQuantizer(
        n_bins=32, warm_start=10
    )
)
metric = metrics.Accuracy()

evaluate.progressive_val_score(dataset, model, metric)
```

```text
Accuracy: 82.24%
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
> - **w** — defaults to `1.0`

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
> - **x** — `dict`
> - **kwargs** — `Any`
>
> **Returns**
>
> `dict&#91;base.typing.ClfTarget, float&#93;`: A dictionary that associates a probability which each label.

[^/dev/api/tree/SGTClassifier/#1]: Gouk, H., Pfahringer, B., &amp; Frank, E. (2019, October). Stochastic Gradient Trees. In Asian Conference on Machine Learning (pp. 1094-1109).
