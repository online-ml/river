# FMRegressor

Factorization Machine for regression.

The model equation is defined as:

\\\[\\hat{y}(x) = w\_{0} + \\sum\_{j=1}^{p} w\_{j} x\_{j} + \\sum\_{j=1}^{p} \\sum\_{j'=j+1}^{p} \\langle \\mathbf{v}\_j, \\mathbf{v}\_{j'} \\rangle x\_{j} x\_{j'}\\\]

Where \\(\\mathbf{v}\_j\\) and \\(\\mathbf{v}\_{j'}\\) are \\(j\\) and \\(j'\\) latent vectors, respectively.

For more efficiency, this model automatically one-hot encodes strings features considering them as categorical variables.

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
    ({'user': 'Alice', 'item': 'Superman'}, 8),
    ({'user': 'Alice', 'item': 'Terminator'}, 9),
    ({'user': 'Alice', 'item': 'Star Wars'}, 8),
    ({'user': 'Alice', 'item': 'Notting Hill'}, 2),
    ({'user': 'Alice', 'item': 'Harry Potter '}, 5),
    ({'user': 'Bob', 'item': 'Superman'}, 8),
    ({'user': 'Bob', 'item': 'Terminator'}, 9),
    ({'user': 'Bob', 'item': 'Star Wars'}, 8),
    ({'user': 'Bob', 'item': 'Notting Hill'}, 2)
)

model = facto.FMRegressor(
    n_factors=10,
    intercept=5,
    seed=42,
)

for x, y in dataset:
    model.learn_one(x, y)

model.predict_one({'Bob': 1, 'Harry Potter': 1})
```

```text
5.236504
```

```python
report = model.debug_one({'Bob': 1, 'Harry Potter': 1})

print(report)
```

```text
Name                 Value      Weight     Contribution
         Intercept    1.00000    5.23426        5.23426
Bob - Harry Potter    1.00000    0.00224        0.00224
      Harry Potter    1.00000    0.00000        0.00000
               Bob    1.00000    0.00000        0.00000
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

- [Rendle, S., 2010, December. Factorization machines. In 2010 IEEE International Conference on Data Mining (pp. 995-1000). IEEE.](<https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf>)
- [Rendle, S., 2012, May. Factorization Machines with libFM. In ACM Transactions on Intelligent Systems and Technology 3, 3, Article 57, 22 pages.](<https://analyticsconsultores.com.mx/wp-content/uploads/2019/03/Factorization-Machines-with-libFM-Steffen-Rendle-University-of-Konstanz2012-.pdf>)
