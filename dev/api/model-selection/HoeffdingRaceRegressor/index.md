# HoeffdingRaceRegressor

Hoeffding Race model selection for regression.

Uses the Hoeffding bound to progressively eliminate underperforming models with statistical confidence. All active models are trained on each observation. After each observation, models whose performance is statistically worse than the current best are eliminated from the race.

This is more efficient than greedy selection when there are many candidate models, because eliminated models no longer consume computation. It is more principled than successive halving because elimination is based on statistical evidence rather than a fixed schedule.

## Parameters

- **models**

  *Type* → `list`

  The models to race against each other.
- **metric**

  *Type* → `metrics.base.RegressionMetric | None`

  *Default* → `None`

  Metric used for comparing models. Defaults to `metrics.MAE`.
- **delta**

  *Type* → `float`

  *Default* → `0.05`

  Confidence parameter. Smaller values make elimination harder (more conservative). The algorithm is correct with probability at least `1 - delta`. Defaults to 0.05.
- **loss\_range**

  *Type* → `float`

  *Default* → `1.0`

  The range of the per-observation loss values. For unbounded metrics you should set this to a reasonable upper bound. Defaults to 1.0.

## Attributes

- **active\_models**

  The models still in the race.
- **best\_model**

  The current best model.
- **models**
- **n\_active\_models**

  The number of models still in the race.

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
    model_selection.HoeffdingRaceRegressor(
        models,
        metric=metric,
        delta=0.05,
        loss_range=50.0,
    )
)

evaluate.progressive_val_score(dataset, model, metric)
```

```text
MAE: 1.319678
```

You can inspect how many models remain in the race:

```python
model["HoeffdingRaceRegressor"].n_active_models
```

```text
4
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

## References

- [Maron, O. and Moore, A.W., 1993. Hoeffding Races: Accelerating Model Selection Search for Classification and Function Approximation. Advances in Neural Information Processing Systems, 6.](<https://papers.nips.cc/paper/1993/hash/02a32ad2669e6fe298e607fe7cc0e1a0-Abstract.html>)
