# PARegressor

Passive-aggressive learning for regression.

## Parameters

- **C**

  *Type* → `float`

  *Default* → `1.0`
- **mode**

  *Type* → `int`

  *Default* → `1`
- **eps**

  *Type* → `float`

  *Default* → `0.1`
- **learn\_intercept**

  *Type* → `bool`

  *Default* → `True`

## Examples

The following example is taken from [this blog post](<https://www.bonaccorso.eu/2017/10/06/ml-algorithms-addendum-passive-aggressive-algorithms/>).

```python
from river import linear_model
from river import metrics
from river import stream
import numpy as np
from sklearn import datasets

np.random.seed(1000)
X, y = datasets.make_regression(n_samples=500, n_features=4)

model = linear_model.PARegressor(
    C=0.01,
    mode=2,
    eps=0.1,
    learn_intercept=False
)
metric = metrics.MAE() + metrics.MSE()

for xi, yi in stream.iter_array(X, y):
    y_pred = model.predict_one(xi)
    model.learn_one(xi, yi)
    metric.update(yi, y_pred)

print(metric)
```

```text
MAE: 9.809402
MSE: 472.393532
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
> - **x** — `dict[base.typing.FeatureName, typing.Any]`
> - **y** — `base.typing.RegTarget`

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

## References

- [Crammer, K., Dekel, O., Keshet, J., Shalev-Shwartz, S. and Singer, Y., 2006. Online passive-aggressive algorithms. Journal of Machine Learning Research, 7(Mar), pp.551-585.](<http://jmlr.csail.mit.edu/papers/volume7/crammer06a/crammer06a.pdf>)
