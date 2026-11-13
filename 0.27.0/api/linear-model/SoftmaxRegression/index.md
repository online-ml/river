# SoftmaxRegression

Softmax regression is a generalization of logistic regression to multiple classes.

Softmax regression is also known as "multinomial logistic regression". There are a set weights for each class, hence the `weights` attribute is a nested `collections.defaultdict`. The main advantage of using this instead of a one-vs-all logistic regression is that the probabilities will be calibrated. Moreover softmax regression is more robust to outliers.

## Parameters

- **optimizer**

  *Type* → `optim.base.Optimizer | None`

  *Default* → `None`

  The sequential optimizer used to tune the weights.
- **loss**

  *Type* → `optim.losses.MultiClassLoss | None`

  *Default* → `None`

  The loss function to optimize for.
- **l2**

  *Type* → `float`

  *Default* → `0`

  Amount of L2 regularization used to push weights towards 0.

## Attributes

- **weights** (`collections.defaultdict`)

## Examples

```python
from river import datasets
from river import evaluate
from river import linear_model
from river import metrics
from river import optim
from river import preprocessing

dataset = datasets.ImageSegments()

model = preprocessing.StandardScaler()
model |= linear_model.SoftmaxRegression()

metric = metrics.MacroF1()

evaluate.progressive_val_score(dataset, model, metric)
```

```text
MacroF1: 81.88%
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
> - **x** — `dict[base.typing.FeatureName, typing.Any]`
> - **kwargs** — `typing.Any`
>
> **Returns**
>
> `dict&#91;base.typing.ClfTarget, float&#93;`: A dictionary that associates a probability which each label.

## References

- [Course on classification stochastic gradient descent](<https://www.inf.ed.ac.uk/teaching/courses/mlp/2016/mlp02-sln.pdf>)
- [Binary vs. Multi-Class Logistic Regression](<https://chrisyeh96.github.io/2018/06/11/logistic-regression.html>)
