# LogisticRegression

Logistic regression.

This estimator supports learning with mini-batches. On top of the single instance methods, it provides the following methods: `learn_many`, `predict_many`, `predict_proba_many`. Each method takes as input a `pandas.DataFrame` where each column represents a feature.

It is generally a good idea to scale the data beforehand in order for the optimizer to converge. You can do this online with a [`preprocessing.StandardScaler`](<https://riverml.xyz/dev/api/preprocessing/StandardScaler>).

## Parameters

- **optimizer** *(mutable)*

  *Type* → `optim.base.Optimizer | None`

  *Default* → `None`

  The sequential optimizer used for updating the weights. Note that the intercept is handled separately.
- **loss** *(mutable)*

  *Type* → `optim.losses.BinaryLoss | None`

  *Default* → `None`

  The loss function to optimize for. Defaults to `optim.losses.Log`.
- **l2** *(mutable)*

  *Type* → `float`

  *Default* → `0.0`

  Amount of L2 regularization used to push weights towards 0. For now, only one type of penalty can be used. The joint use of L1 and L2 is not explicitly supported.
- **l1** *(mutable)*

  *Type* → `float`

  *Default* → `0.0`

  Amount of L1 regularization used to push weights towards 0. For now, only one type of penalty can be used. The joint use of L1 and L2 is not explicitly supported.
- **intercept\_init**

  *Type* → `float`

  *Default* → `0.0`

  Initial intercept value.
- **intercept\_lr** *(mutable)*

  *Type* → `float | optim.base.Scheduler`

  *Default* → `0.01`

  Learning rate scheduler used for updating the intercept. A `optim.schedulers.Constant` is used if a `float` is provided. The intercept is not updated when this is set to 0.
- **clip\_gradient** *(mutable)*

  *Type* → `float`

  *Default* → `1000000000000.0`

  Clips the absolute value of each gradient value.
- **initializer** *(mutable)*

  *Type* → `optim.base.Initializer | None`

  *Default* → `None`

  Weights initialization scheme.

## Attributes

- **weights**

  The current weights.

## Examples

```python
from river import datasets
from river import evaluate
from river import linear_model
from river import metrics
from river import optim
from river import preprocessing

dataset = datasets.Phishing()

model = (
    preprocessing.StandardScaler() |
    linear_model.LogisticRegression(optimizer=optim.SGD(.1))
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
