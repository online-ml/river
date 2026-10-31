# BaggingRegressor

Online bootstrap aggregation for regression.

For each incoming observation, each model's `learn_one` method is called `k` times where `k` is sampled from a Poisson distribution of parameter 1. `k` thus has a 36% chance of being equal to 0, a 36% chance of being equal to 1, an 18% chance of being equal to 2, a 6% chance of being equal to 3, a 1% chance of being equal to 4, etc. You can do `scipy.stats.utils.random.poisson(1).pmf(k)` for more detailed values.

## Parameters

- **model**

  *Type* → `base.Regressor`

  The regressor to bag.
- **n\_models**

  *Default* → `10`

  The number of models in the ensemble.
- **seed**

  *Type* → `int | None`

  *Default* → `None`

  Random number generator seed for reproducibility.

## Attributes

- **models**

## Examples

In the following example three logistic regressions are bagged together. The performance is slightly better than when using a single logistic regression.

```python
from river import datasets
from river import ensemble
from river import evaluate
from river import linear_model
from river import metrics
from river import optim
from river import preprocessing

dataset = datasets.TrumpApproval()

model = preprocessing.StandardScaler()
model |= ensemble.BaggingRegressor(
    model=linear_model.LinearRegression(intercept_lr=0.1),
    n_models=3,
    seed=42
)

metric = metrics.MAE()

evaluate.progressive_val_score(dataset, model, metric)
```

```text
MAE: 0.677586
```

## Methods

> [!NOTE]
>
> **learn\_one**

> [!NOTE]
>
> **predict\_one**
>
> Averages the predictions of each regressor.
>
> **Parameters**
>
> - **x**
> - **kwargs**

## References

- [Oza, N.C., 2005, October. Online bagging and boosting. In 2005 IEEE international conference on systems, man and cybernetics (Vol. 3, pp. 2340-2345). Ieee.](<https://ti.arc.nasa.gov/m/profile/oza/files/ozru01a.pdf>)
