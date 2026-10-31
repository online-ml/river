# LinearRegression

Linear regression.

This estimator supports learning with mini-batches. On top of the single instance methods, it provides the following methods: `learn_many`, `predict_many`, `predict_proba_many`. Each method takes as input a `pandas.DataFrame` where each column represents a feature.

It is generally a good idea to scale the data beforehand in order for the optimizer to converge. You can do this online with a [`preprocessing.StandardScaler`](<https://riverml.xyz/dev/api/preprocessing/StandardScaler>).

## Parameters

- **optimizer** *(mutable)*

  *Type* → `optim.base.Optimizer | None`

  *Default* → `None`

  The sequential optimizer used for updating the weights. Note that the intercept updates are handled separately.
- **loss** *(mutable)*

  *Type* → `optim.losses.RegressionLoss | None`

  *Default* → `None`

  The loss function to optimize for.
- **l2** *(mutable)*

  *Type* → `float`

  *Default* → `0.0`

  Amount of L2 regularization used to push weights towards 0. For now, only one type of penalty can be used. The joint use of L1 and L2 is not explicitly supported.
- **l1** *(mutable)*

  *Type* → `float`

  *Default* → `0.0`

  Amount of L1 regularization used to push weights towards 0. For now, only one type of penalty can be used. The joint use of L1 and L2 is not explicitly supported.
- **intercept\_init**

  *Type* → `float`

  *Default* → `0.0`

  Initial intercept value.
- **intercept\_lr** *(mutable)*

  *Type* → `optim.base.Scheduler | float`

  *Default* → `0.01`

  Learning rate scheduler used for updating the intercept. A `optim.schedulers.Constant` is used if a `float` is provided. The intercept is not updated when this is set to 0.
- **clip\_gradient** *(mutable)*

  *Type* → `float`

  *Default* → `1000000000000.0`

  Clips the absolute value of each gradient value.
- **initializer** *(mutable)*

  *Type* → `optim.base.Initializer | None`

  *Default* → `None`

  Weights initialization scheme.

## Attributes

- **weights** (`dict`)

  The current weights.

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
    linear_model.LinearRegression(intercept_lr=.1)
)
metric = metrics.MAE()

evaluate.progressive_val_score(dataset, model, metric)
```

```text
MAE: 0.558735
```

```python
model['LinearRegression'].intercept
```

```text
35.617670
```

You can call the `debug_one` method to break down a prediction. This works even if the linear regression is part of a pipeline.

```python
x, y = next(iter(dataset))
report = model.debug_one(x)
print(report)
```

```text
0. Input
--------
gallup: 43.84321 (float)
ipsos: 46.19925 (float)
morning_consult: 48.31875 (float)
ordinal_date: 736389 (int)
rasmussen: 44.10469 (float)
you_gov: 43.63691 (float)
<BLANKLINE>
1. StandardScaler
-----------------
gallup: 1.18810 (float)
ipsos: 2.10348 (float)
morning_consult: 2.73545 (float)
ordinal_date: -1.73032 (float)
rasmussen: 1.26872 (float)
you_gov: 1.48391 (float)
<BLANKLINE>
2. LinearRegression
-------------------
Name              Value      Weight      Contribution
      Intercept    1.00000    35.61767       35.61767
          ipsos    2.10348     0.62689        1.31866
morning_consult    2.73545     0.24180        0.66144
         gallup    1.18810     0.43568        0.51764
      rasmussen    1.26872     0.28118        0.35674
        you_gov    1.48391     0.03123        0.04634
   ordinal_date   -1.73032     3.45162       -5.97242
<BLANKLINE>
Prediction: 32.54607
```

## Methods

> [!NOTE]
>
> **debug\_one**
>
> Debugs the output of the linear regression.
>
> **Parameters**
>
> - **x** — `dict[base.typing.FeatureName, typing.Any]`
> - **decimals** — `int` — defaults to `5`
>
> **Returns**
>
> `str`: A table which explains the output.

> [!NOTE]
>
> **learn\_many**
>
> Update the model with a mini-batch of features `X` and real-valued targets `y`.
>
> **Parameters**
>
> - **X** — `IntoDataFrame`
> - **y** — `IntoSeries`
> - **w** — `float | IntoSeries` — defaults to `1`

> [!NOTE]
>
> **learn\_one**
>
> Fits to a set of features `x` and a real-valued target `y`.
>
> **Parameters**
>
> - **x** — `dict[base.typing.FeatureName, typing.Any]`
> - **y** — `typing.Any`
> - **w** — `float` — defaults to `1.0`

> [!NOTE]
>
> **predict\_many**
>
> Predict the outcome for each given sample.
>
> **Parameters**
>
> - **X** — `IntoDataFrame`
>
> **Returns**
>
> `IntoSeries`: The predicted outcomes.

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
