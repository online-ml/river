# HOFMClassifier

Higher-Order Factorization Machine for binary classification.

The model equation is defined as:

\\\[\\hat{y}(x) = w\_{0} + \\sum\_{j=1}^{p} w\_{j} x\_{j} + \\sum\_{l=2}^{d} \\sum\_{j\_1=1}^{p} \\cdots \\sum\_{j\_l=j\_{l-1}+1}^{p} \\left(\\prod\_{j'=1}^{l} x\_{j\_{j'}} \\right) \\left(\\sum\_{f=1}^{k\_l} \\prod\_{j'=1}^{l} v\_{j\_{j'}, f}^{(l)} \\right)\\\]

For more efficiency, this model automatically one-hot encodes strings features considering them as categorical variables.

## Parameters

- **degree**

  *Default* → `3`

  Polynomial degree or model order.
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

  *Type* → `optim.losses.BinaryLoss | None`

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
    ({'user': 'Alice', 'item': 'Superman', 'time': .12}, True),
    ({'user': 'Alice', 'item': 'Terminator', 'time': .13}, True),
    ({'user': 'Alice', 'item': 'Star Wars', 'time': .14}, True),
    ({'user': 'Alice', 'item': 'Notting Hill', 'time': .15}, False),
    ({'user': 'Alice', 'item': 'Harry Potter ', 'time': .16}, True),
    ({'user': 'Bob', 'item': 'Superman', 'time': .13}, True),
    ({'user': 'Bob', 'item': 'Terminator', 'time': .12}, True),
    ({'user': 'Bob', 'item': 'Star Wars', 'time': .16}, True),
    ({'user': 'Bob', 'item': 'Notting Hill', 'time': .10}, False)
)

model = facto.HOFMClassifier(
    degree=3,
    n_factors=10,
    intercept=.5,
    seed=42,
)

for x, y in dataset:
    model.learn_one(x, y)

model.predict_one({'user': 'Bob', 'item': 'Harry Potter', 'time': .14})
```

```text
True
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
> Update the model with a set of features `x` and a label `y`.
>
> **Parameters**
>
> - **x** — `dict[base.typing.FeatureName, Any]`
> - **y** — `base.typing.ClfTarget`
> - **w** — defaults to `1.0`

> [!NOTE]
>
> **predict\_one**
>
> Predict the label of a set of features `x`.
>
> **Parameters**
>
> - **x** — `dict[base.typing.FeatureName, Any]`
> - **kwargs** — `Any`
>
> **Returns**
>
> `base.typing.ClfTarget | None`: The predicted label.

> [!NOTE]
>
> **predict\_proba\_one**
>
> Predict the probability of each label for a dictionary of features `x`.
>
> **Parameters**
>
> - **x**
>
> **Returns**
>
> A dictionary that associates a probability which each label.

## References

- [Rendle, S., 2010, December. Factorization machines. In 2010 IEEE International Conference on Data Mining (pp. 995-1000). IEEE.](<https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf>)
