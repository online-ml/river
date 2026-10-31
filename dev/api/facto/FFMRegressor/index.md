# FFMRegressor

Field-aware Factorization Machine for regression.

The model equation is defined by:

\\\[\\hat{y}(x) = w\_{0} + \\sum\_{j=1}^{p} w\_{j} x\_{j} + \\sum\_{j=1}^{p} \\sum\_{j'=j+1}^{p} \\langle \\mathbf{v}\_{j, f\_{j'}}, \\mathbf{v}\_{j', f\_j} \\rangle x\_{j} x\_{j'}\\\]

Where \\(\\mathbf{v}\_{j, f\_{j'}}\\) is the latent vector corresponding to \\(j\\) feature for \\(f\_{j'}\\) field, and \\(\\mathbf{v}\_{j', f\_j}\\) is the latent vector corresponding to \\(j'\\) feature for \\(f\_j\\) field.

For more efficiency, this model automatically one-hot encodes strings features considering them as categorical variables. Field names are inferred from feature names by taking everything before the first underscore: `feature_name.split('_')\[0\]`.

## Parameters

- **n\_factors**

  *Default* → `10`

  Dimensionality of the factorization or number of latent factors.
- **weight\_optimizer**

  *Type* → `optim.base.Optimizer | None`

  *Default* → `None`

  The sequential optimizer used for updating the feature weights. Note that the intercept is handled separately.
- **latent\_optimizer**

  *Type* → `optim.base.Optimizer | None`

  *Default* → `None`

  The sequential optimizer used for updating the latent factors.
- **loss**

  *Type* → `optim.losses.RegressionLoss | None`

  *Default* → `None`

  The loss function to optimize for.
- **sample\_normalization**

  *Default* → `False`

  Whether to divide each element of `x` by `x`'s L2-norm.
- **l1\_weight**

  *Default* → `0.0`

  Amount of L1 regularization used to push weights towards 0.
- **l2\_weight**

  *Default* → `0.0`

  Amount of L2 regularization used to push weights towards 0.
- **l1\_latent**

  *Default* → `0.0`

  Amount of L1 regularization used to push latent weights towards 0.
- **l2\_latent**

  *Default* → `0.0`

  Amount of L2 regularization used to push latent weights towards 0.
- **intercept**

  *Default* → `0.0`

  Initial intercept value.
- **intercept\_lr**

  *Type* → `optim.base.Scheduler | float`

  *Default* → `0.01`

  Learning rate scheduler used for updating the intercept. An instance of `optim.schedulers.Constant` is used if a `float` is passed. No intercept will be used if this is set to 0.
- **weight\_initializer**

  *Type* → `optim.initializers.Initializer | None`

  *Default* → `None`

  Weights initialization scheme. Defaults to `optim.initializers.Zeros()`.
- **latent\_initializer**

  *Type* → `optim.initializers.Initializer | None`

  *Default* → `None`

  Latent factors initialization scheme. Defaults to `optim.initializers.Normal(mu=.0, sigma=.1, random_state=self.random_state)`.
- **clip\_gradient**

  *Default* → `1000000000000.0`

  Clips the absolute value of each gradient value.
- **seed**

  *Type* → `int | None`

  *Default* → `None`

  Randomization seed used for reproducibility.

## Attributes

- **weights**

  The current weights assigned to the features.
- **latents**

  The current latent weights assigned to the features.

## Examples

```python
from river import facto

dataset = (
    ({'user': 'Alice', 'item': 'Superman', 'time': .12}, 8),
    ({'user': 'Alice', 'item': 'Terminator', 'time': .13}, 9),
    ({'user': 'Alice', 'item': 'Star Wars', 'time': .14}, 8),
    ({'user': 'Alice', 'item': 'Notting Hill', 'time': .15}, 2),
    ({'user': 'Alice', 'item': 'Harry Potter ', 'time': .16}, 5),
    ({'user': 'Bob', 'item': 'Superman', 'time': .13}, 8),
    ({'user': 'Bob', 'item': 'Terminator', 'time': .12}, 9),
    ({'user': 'Bob', 'item': 'Star Wars', 'time': .16}, 8),
    ({'user': 'Bob', 'item': 'Notting Hill', 'time': .10}, 2)
)

model = facto.FFMRegressor(
    n_factors=10,
    intercept=5,
    seed=42,
)

for x, y in dataset:
    model.learn_one(x, y)

model.predict_one({'user': 'Bob', 'item': 'Harry Potter', 'time': .14})
```

```text
5.319945
```

```python
report = model.debug_one({'user': 'Bob', 'item': 'Harry Potter', 'time': .14})

print(report)
```

```text
Name                                       Value      Weight     Contribution
                               Intercept    1.00000    5.23501        5.23501
                                user_Bob    1.00000    0.11438        0.11438
                                    time    0.14000    0.03186        0.00446
    item_Harry Potter(time) - time(item)    0.14000    0.03153        0.00441
             user_Bob(time) - time(user)    0.14000    0.02864        0.00401
                       item_Harry Potter    1.00000    0.00000        0.00000
user_Bob(item) - item_Harry Potter(user)    1.00000   -0.04232       -0.04232
```

## Methods

> [!NOTE]
>
> **debug\_one**
>
> Debugs the output of the FM regressor.
>
> **Parameters**
>
> - **x** — `dict`
> - **decimals** — `int` — defaults to `5`
>
> **Returns**
>
> `str`: A table which explains the output.

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
> - **w** — defaults to `1.0`

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

- [Juan, Y., Zhuang, Y., Chin, W.S. and Lin, C.J., 2016, September. Field-aware factorization machines for CTR prediction. In Proceedings of the 10th ACM Conference on Recommender Systems (pp. 43-50).](<https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf>)
