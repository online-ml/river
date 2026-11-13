# RegressorChain

A multi-output model that arranges regressors into a chain.

This will create one model per output. The prediction of the first output will be used as a feature in the second output. The prediction for the second output will be used as a feature for the third, etc. This "chain model" is therefore capable of capturing dependencies between outputs.

## Parameters

- **model**

  *Type* → `base.Regressor`

  The regression model used to make predictions for each target.
- **order**

  *Type* → `list | None`

  *Default* → `None`

  A list with the targets order in which to construct the chain. If `None` then the order will be inferred from the order of the keys in the target.

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

model = multioutput.RegressorChain(
    model=(
        preprocessing.StandardScaler() |
        linear_model.LinearRegression(intercept_lr=0.3)
    ),
    order=[0, 1, 2]
)

metric = metrics.multioutput.MicroAverage(metrics.MAE())

evaluate.progressive_val_score(dataset, model, metric)
```

```text
MicroAverage(MAE): 12.733525
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
