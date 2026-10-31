# BanditRegressor

Bandit-based model selection for regression.

Each model is associated with an arm. At each `learn_one` call, the policy decides which arm/model to pull. The reward is the performance of the model on the provided sample. The `predict_one` method uses the current best model.

## Parameters

- **models**

  The models to select from.
- **metric**

  *Type* → `metrics.base.RegressionMetric`

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
    linear_model.LinearRegression(optimizer=optim.SGD(lr=lr))
    for lr in [0.0001, 0.001, 1e-05, 0.01]
]

dataset = datasets.TrumpApproval()
model = (
    preprocessing.StandardScaler() |
    model_selection.BanditRegressor(
        models,
        metric=metrics.MAE(),
        policy=bandit.EpsilonGreedy(
            epsilon=0.1,
            decay=0.001,
            burn_in=100,
            seed=42
        )
    )
)
metric = metrics.MAE()

evaluate.progressive_val_score(dataset, model, metric)
```

```text
MAE: 3.134089
```

Here's another example using the UCB policy. The latter is more sensitive to the target scale, and usually works better when the target is rescaled.

```python
models = [
    linear_model.LinearRegression(optimizer=optim.SGD(lr=lr))
    for lr in [0.0001, 0.001, 1e-05, 0.01]
]

model = (
    preprocessing.StandardScaler() |
    preprocessing.TargetStandardScaler(
        model_selection.BanditRegressor(
            models,
            metric=metrics.MAE(),
            policy=bandit.UCB(
                delta=1,
                burn_in=100
            )
        )
    )
)
metric = metrics.MAE()

evaluate.progressive_val_score(dataset, model, metric)
```

```text
MAE: 0.875333
```

## Methods

> [!NOTE]
>
> **learn\_one**
>
> Fits to a set of features `x` and a real-valued target `y`.
>
> **Parameters**
>
> - **x**
> - **y**

> [!NOTE]
>
> **predict\_one**
>
> Predict the output of features `x`.
>
> **Parameters**
>
> - **x**
>
> **Returns**
>
> The prediction.
