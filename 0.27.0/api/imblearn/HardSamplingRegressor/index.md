# HardSamplingRegressor

Hard sampling regressor.

This wrapper enables a model to retrain on past samples who's output was hard to predict. This works by storing the hardest samples in a buffer of a fixed size. When a new sample arrives, the wrapped model is either trained on one of the buffered samples with a probability p or on the new sample with a probability (1 - p).

The hardness of an observation is evaluated with a loss function that compares the sample's ground truth with the wrapped model's prediction. If the buffer is not full, then the sample is added to the buffer. If the buffer is full and the new sample has a bigger loss than the lowest loss in the buffer, then the sample takes its place.

## Parameters

- **regressor**

  *Type* → `base.Regressor`
- **size**

  *Type* → `int`

  Size of the buffer.
- **p**

  *Type* → `float`

  Probability of updating the model with a sample from the buffer instead of a new incoming sample.
- **loss**

  *Type* → `optim.losses.RegressionLoss | None`

  *Default* → `None`

  Criterion used to evaluate the hardness of a sample.
- **seed**

  *Type* → `int | None`

  *Default* → `None`

  Random seed.

## Attributes

- **regressor**

## Examples

```python
from river import datasets
from river import evaluate
from river import imblearn
from river import linear_model
from river import metrics
from river import optim
from river import preprocessing

model = (
    preprocessing.StandardScaler() |
    imblearn.HardSamplingRegressor(
        regressor=linear_model.LinearRegression(),
        p=.2,
        size=30,
        seed=42,
    )
)

evaluate.progressive_val_score(
    datasets.TrumpApproval(),
    model,
    metrics.MAE(),
    print_every=500
)
```

```text
[500] MAE: 2.274021
[1,000] MAE: 1.392399
[1,001] MAE: 1.391246
MAE: 1.391246
```

## Methods

> [!NOTE]
>
> **learn\_one**

> [!NOTE]
>
> **predict\_one**
>
> Predict the output of features `x`.
>
> **Parameters**
>
> - **x** — `dict[base.typing.FeatureName, typing.Any]`
> - **kwargs** — `typing.Any`
>
> **Returns**
>
> `base.typing.RegTarget`: The prediction.
