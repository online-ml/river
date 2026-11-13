# TargetMinMaxScaler

Applies min-max scaling to the target.

## Parameters

- **regressor**

  *Type* → `base.Regressor`

  Regression model to wrap.

## Examples

```python
from river import datasets
from river import evaluate
from river import linear_model
from river import metrics
from river import preprocessing

dataset = datasets.TrumpApproval()
model = (
    preprocessing.StandardScaler() |
    preprocessing.TargetMinMaxScaler(
        regressor=linear_model.LinearRegression(intercept_lr=0.15)
    )
)
metric = metrics.MSE()

evaluate.progressive_val_score(dataset, model, metric)
```

```text
MSE: 2.018905
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
