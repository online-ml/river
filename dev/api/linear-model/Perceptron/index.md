# Perceptron

Perceptron classifier.

In this implementation, the Perceptron is viewed as a special case of the logistic regression. The loss function that is used is the Hinge loss with a threshold set to 0, whilst the learning rate of the stochastic gradient descent procedure is set to 1 for both the weights and the intercept.

## Parameters

- **l2** *(mutable)*

  *Type* → `float`

  *Default* → `0.0`

  Amount of L2 regularization used to push weights towards 0.
- **clip\_gradient** *(mutable)*

  *Type* → `float`

  *Default* → `1000000000000.0`

  Clips the absolute value of each gradient value.
- **initializer** *(mutable)*

  *Type* → `optim.initializers.Initializer | None`

  *Default* → `None`

  Weights initialization scheme.

## Attributes

- **weights**

  The current weights.

## Examples

```python
from river import datasets
from river import evaluate
from river import linear_model as lm
from river import metrics
from river import preprocessing as pp

dataset = datasets.Phishing()

model = pp.StandardScaler() | lm.Perceptron()

metric = metrics.Accuracy()

evaluate.progressive_val_score(dataset, model, metric)
```

```text
Accuracy: 85.84%
```

## Methods

> [!NOTE]
>
> **learn\_many**
>
> Update the model with a mini-batch of features `X` and boolean targets `y`.
>
> **Parameters**
>
> - **X** — `IntoDataFrame`
> - **y** — `IntoSeries`
> - **w** — `float | IntoSeries` — defaults to `1`

> [!NOTE]
>
> **learn\_one**
>
> Update the model with a set of features `x` and a label `y`.
>
> **Parameters**
>
> - **x** — `dict[base.typing.FeatureName, typing.Any]`
> - **y** — `typing.Any`
> - **w** — `float` — defaults to `1.0`

> [!NOTE]
>
> **predict\_many**
>
> Predict the outcome for each given sample.
>
> **Parameters**
>
> - **X** — `IntoDataFrame`
>
> **Returns**
>
> `IntoSeries`: The predicted labels.

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
>
> Predict the outcome probabilities for each given sample.
>
> **Parameters**
>
> - **X** — `IntoDataFrame`
>
> **Returns**
>
> `IntoDataFrame`: A dataframe with probabilities of `True` and `False` for each sample.

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
