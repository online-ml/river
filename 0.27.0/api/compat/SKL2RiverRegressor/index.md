# SKL2RiverRegressor

Compatibility layer from scikit-learn to River for regression.

## Parameters

- **estimator**

  *Type* → `sklearn_base.BaseEstimator`

  A scikit-learn transformer which has a `partial_fit` method.

## Examples

```python
from river import compat
from river import evaluate
from river import metrics
from river import preprocessing
from river import stream
from sklearn import linear_model
from sklearn import datasets

dataset = stream.iter_sklearn_dataset(
    dataset=datasets.load_diabetes(),
    shuffle=True,
    seed=42
)

scaler = preprocessing.StandardScaler()
sgd_reg = compat.convert_sklearn_to_river(linear_model.SGDRegressor())
model = scaler | sgd_reg

metric = metrics.MAE()

evaluate.progressive_val_score(dataset, model, metric)
```

```text
MAE: 84.501421
```

## Methods

> [!NOTE]
>
> **learn\_many**

> [!NOTE]
>
> **learn\_one**
>
> Fits to a set of features `x` and a real-valued target `y`.
>
> **Parameters**
>
> - **x** — `dict[base.typing.FeatureName, typing.Any]`
> - **y** — `base.typing.RegTarget`

> [!NOTE]
>
> **predict\_many**

> [!NOTE]
>
> **predict\_one**
>
> Predict the output of features `x`.
>
> **Parameters**
>
> - **x** — `dict[base.typing.FeatureName, typing.Any]`
>
> **Returns**
>
> `base.typing.RegTarget`: The prediction.
