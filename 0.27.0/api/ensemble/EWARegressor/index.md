# EWARegressor

Exponentially Weighted Average regressor.

## Parameters

- **models**

  *Type* → `list[base.Regressor]`

  The regressors to hedge.
- **loss**

  *Type* → `optim.losses.RegressionLoss | None`

  *Default* → `None`

  The loss function that has to be minimized. Defaults to `optim.losses.Squared`.
- **learning\_rate**

  *Default* → `0.5`

  The learning rate by which the model weights are multiplied at each iteration.

## Attributes

- **models**

## Examples

```python
from river import datasets
from river import ensemble
from river import evaluate
from river import linear_model
from river import metrics
from river import optim
from river import preprocessing
from river import stream

optimizers = [
    optim.SGD(0.01),
    optim.RMSProp(),
    optim.AdaGrad()
]

for optimizer in optimizers:

    dataset = datasets.TrumpApproval()
    metric = metrics.MAE()
    model = (
        preprocessing.StandardScaler() |
        linear_model.LinearRegression(
            optimizer=optimizer,
            intercept_lr=.1
        )
    )

    print(optimizer, evaluate.progressive_val_score(dataset, model, metric))
```

```text
SGD MAE: 0.558735
RMSProp MAE: 0.522449
AdaGrad MAE: 0.477289
```

```python
dataset = datasets.TrumpApproval()
metric = metrics.MAE()
hedge = (
    preprocessing.StandardScaler() |
    ensemble.EWARegressor(
        [
            linear_model.LinearRegression(optimizer=o, intercept_lr=.1)
            for o in optimizers
        ],
        learning_rate=0.005
    )
)

evaluate.progressive_val_score(dataset, hedge, metric)
```

```text
MAE: 0.496298
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
> **learn\_predict\_one**

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

- [Online Learning from Experts: Weighed Majority and Hedge](<https://www.shivani-agarwal.net/Teaching/E0370/Aug-2011/Lectures/20-scribe1.pdf>)
- [Wikipedia page on the multiplicative weight update method](<https://en.wikipedia.org/wiki/Multiplicative_weight_update_method>)
- [Kivinen, J. and Warmuth, M.K., 1997. Exponentiated gradient versus gradient descent for linear predictors. information and computation, 132(1), pp.1-63.](<https://users.soe.ucsc.edu/~manfred/pubs/J36.pdf>)
