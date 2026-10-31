# GreedyRegressor

Greedy selection regressor.

This selection method simply updates each model at each time step. The current best model is used to make predictions. It's greedy in the sense that updating each model can be costly. On the other hand, bandit-like algorithms are more temperate in that only update a subset of the models at each step.

## Parameters

- **models**

  *Type* → `list[base.Regressor]`

  The models to select from.
- **metric**

  *Type* → `metrics.base.RegressionMetric | None`

  *Default* → `None`

  The metric that is used to measure the performance of each model.

## Attributes

- **best\_model**

  The current best model.
- **models**

## Examples

```python
from river import datasets
from river import evaluate
from river import linear_model
from river import metrics
from river import model_selection
from river import optim
from river import preprocessing

models = [
    linear_model.LinearRegression(optimizer=optim.SGD(lr=lr))
    for lr in [1e-5, 1e-4, 1e-3, 1e-2]
]

dataset = datasets.TrumpApproval()
metric = metrics.MAE()
model = (
    preprocessing.StandardScaler() |
    model_selection.GreedyRegressor(models, metric)
)

evaluate.progressive_val_score(dataset, model, metric)
```

```text
MAE: 1.319678
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
> - **x** — `dict[base.typing.FeatureName, Any]`
> - **y** — `base.typing.RegTarget`

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
