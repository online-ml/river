# SuccessiveHalvingRegressor

Successive halving algorithm for regression.

Successive halving is a method for performing model selection without having to train each model on all the dataset. At certain points in time (called "rungs"), the worst performing will be discarded and the best ones will keep competing between each other. The rung values are designed so that at most `budget` model updates will be performed in total.

If you have `k` combinations of hyperparameters and that your dataset contains `n` observations, then the maximal budget you can allocate is:

\\\[\\frac{2kn}{eta}\\\]

It is recommended that you check this beforehand. This bound can't be checked by the function because the size of the dataset is not known. In fact it is potentially infinite, in which case the algorithm will terminate once all the budget has been spent.

If you have a budget of `B`, and that your dataset contains `n` observations, then the number of hyperparameter combinations that will spend all the budget and go through all the data is:

\\\[\\left\\lceil\\left\\lfloor\\frac{B}{2n}\\right\\rfloor \\times eta \\right\\rceil\\\]

## Parameters

- **models**

  The models to compare.
- **metric**

  *Type* → `metrics.base.Metric`

  Metric used for comparing models with.
- **budget**

  *Type* → `int`

  Total number of model updates you wish to allocate.
- **eta**

  *Default* → `2`

  Rate of elimination. At every rung, `math.ceil(k / eta)` models are kept, where `k` is the number of models that have reached the rung. A higher `eta` value will focus on less models but will allocate more iterations to the best models.
- **verbose**

  *Default* → `False`

  Whether to display progress or not.

## Attributes

- **best\_model**

  The current best model.
- **models**

## Examples

As an example, let's use successive halving to tune the optimizer of a linear regression. We'll first define the model.

```python
from river import linear_model
from river import preprocessing

model = (
    preprocessing.StandardScaler() |
    linear_model.LinearRegression(intercept_lr=.1)
)
```

Let's now define a grid of parameters which we would like to compare. We'll try different optimizers with various learning rates.

```python
from river import optim
from river import utils

models = utils.expand_param_grid(model, {
    'LinearRegression': {
        'optimizer': [
            (optim.SGD, {'lr': [.1, .01, .005]}),
            (optim.Adam, {'beta_1': [.01, .001], 'lr': [.1, .01, .001]}),
            (optim.Adam, {'beta_1': [.1], 'lr': [.001]}),
        ]
    }
})
```

We can check how many models we've created.

```python
len(models)
```

```text
10
```

We can now pass these models to a `SuccessiveHalvingRegressor`. We also need to pick a metric to compare the models, and a budget which indicates how many iterations to run before picking the best model and discarding the rest.

```python
from river import model_selection

sh = model_selection.SuccessiveHalvingRegressor(
    models,
    metric=metrics.MAE(),
    budget=2000,
    eta=2,
    verbose=True
)
```

A `SuccessiveHalvingRegressor` is also a regressor with a `learn_one` and a `predict_one` method. We can therefore evaluate it like any other classifier with [`evaluate.progressive_val_score`](<https://riverml.xyz/dev/api/evaluate/progressive-val-score>).

```python
from river import datasets
from river import evaluate
from river import metrics

evaluate.progressive_val_score(
    dataset=datasets.TrumpApproval(),
    model=sh,
    metric=metrics.MAE()
)
```

```text
[1] 5 removed       5 left  50 iterations   budget used: 500        budget left: 1500       best MAE: 4.419643
[2] 2 removed       3 left  100 iterations  budget used: 1000       budget left: 1000       best MAE: 2.392266
[3] 1 removed       2 left  166 iterations  budget used: 1498       budget left: 502        best MAE: 1.541383
[4] 1 removed       1 left  250 iterations  budget used: 1998       budget left: 2  best MAE: 1.112122
MAE: 0.490688
```

We can now view the best model.

```python
sh.best_model
```

```text
Pipeline (
  StandardScaler (
    with_std=True
    window_size=None
  ),
  LinearRegression (
    optimizer=Adam (
      lr=Constant (
        learning_rate=0.1
      )
      beta_1=0.01
      beta_2=0.999
      eps=1e-08
    )
    loss=Squared ()
    l2=0.
    l1=0.
    intercept_init=0.
    intercept_lr=Constant (
      learning_rate=0.1
    )
    clip_gradient=1e+12
    initializer=Zeros ()
  )
)
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
> - **x** — `dict[base.typing.FeatureName, Any]`
> - **y** — `base.typing.RegTarget`

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

- [Jamieson, K. and Talwalkar, A., 2016, May. Non-stochastic best arm identification and hyperparameter optimization. In Artificial Intelligence and Statistics (pp. 240-248).](<http://proceedings.mlr.press/v51/jamieson16.pdf>)
- [Li, L., Jamieson, K., Rostamizadeh, A., Gonina, E., Hardt, M., Recht, B. and Talwalkar, A., 2018. Massively parallel hyperparameter tuning. arXiv preprint arXiv:1810.05934.](<https://arxiv.org/pdf/1810.05934.pdf>)
- [Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A. and Talwalkar, A., 2017. Hyperband: A novel bandit-based approach to hyperparameter optimization. The Journal of Machine Learning Research, 18(1), pp.6765-6816.](<https://arxiv.org/pdf/1603.06560.pdf>)
