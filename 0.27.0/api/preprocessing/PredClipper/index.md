# PredClipper

Clips the target after predicting.

## Parameters

- **regressor**

  *Type* → `base.Regressor`

  Regressor model for which to clip the predictions.
- **y\_min**

  *Type* → `float`

  minimum value.
- **y\_max**

  *Type* → `float`

  maximum value.

## Examples

```python
from river import linear_model
from river import preprocessing

dataset = (
    ({'a': 2, 'b': 4}, 80),
    ({'a': 3, 'b': 5}, 100),
    ({'a': 4, 'b': 6}, 120)
)

model = preprocessing.PredClipper(
    regressor=linear_model.LinearRegression(),
    y_min=0,
    y_max=200
)

for x, y in dataset:
    model.learn_one(x, y)

model.predict_one({'a': -100, 'b': -200})
```

```text
0
```

```python
model.predict_one({'a': 50, 'b': 60})
```

```text
200
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
> Predict the output of features `x`.
>
> **Parameters**
>
> - **x**
> - **kwargs**
>
> **Returns**
>
> The prediction.
