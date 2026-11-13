# TargetTransformRegressor

Modifies the target before training.

The user is expected to check that `func` and `inverse_func` are coherent with each other.

## Parameters

- **regressor**

  *Type* → `base.Regressor`

  Regression model to wrap.
- **func**

  *Type* → `typing.Callable`

  A function modifying the target before training.
- **inverse\_func**

  *Type* → `typing.Callable`

  A function to return to the target's original space.

## Examples

```python
import math
from river import compose
from river import datasets
from river import evaluate
from river import linear_model
from river import metrics
from river import preprocessing

dataset = datasets.TrumpApproval()
model = (
    preprocessing.StandardScaler() |
    compose.TargetTransformRegressor(
        regressor=linear_model.LinearRegression(intercept_lr=0.15),
        func=math.log,
        inverse_func=math.exp
    )
)
metric = metrics.MSE()

evaluate.progressive_val_score(dataset, model, metric)
```

```text
MSE: 10.999752
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
