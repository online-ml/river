# PerOutputRegressor

A multi-output model that trains one independent regressor per output.

This model does not use the prediction of one output as a feature for the next. Each output is modelled by its own copy of the base regressor, trained independently. (This is the streaming equivalent of scikit-learn's `MultiOutputRegressor`).

The set of outputs isn't known from the start in a streaming setting, new regressors are instantiated on the fly, one per output key encountered in `y`.

## Parameters

- **model**

  *Type* → `base.Regressor`

  The regression model used to make predictions for each target.

## Examples

```python
from river import evaluate
from river import linear_model
from river import metrics
from river import multioutput
from river import preprocessing
from river import stream

from sklearn import datasets

dataset = stream.iter_sklearn_dataset(
    dataset=datasets.load_linnerud(),
    shuffle=True,
    seed=42
)

model = multioutput.PerOutputRegressor(
    model=(
        preprocessing.StandardScaler() |
        linear_model.LinearRegression(intercept_lr=0.3)
    )
)

metric = metrics.multioutput.MicroAverage(metrics.MAE())

evaluate.progressive_val_score(dataset, model, metric)
```

```text
MicroAverage(MAE): 12.68377
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
> - **kwargs**

> [!NOTE]
>
> **predict\_one**
>
> Predict the outputs of features `x`.
>
> **Parameters**
>
> - **x**
> - **kwargs**
>
> **Returns**
>
> The predictions.
