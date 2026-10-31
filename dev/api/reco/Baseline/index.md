# Baseline

Baseline for recommender systems.

A first-order approximation of the bias involved in target. The model equation is defined as:

\\\[\\hat{y}(x) = \\bar{y} + bu\_{u} + bi\_{i}\\\]

Where \\(bu\_{u}\\) and \\(bi\_{i}\\) are respectively the user and item biases.

This model expects a dict input with a `user` and an `item` entries without any type constraint on their values (i.e. can be strings or numbers). Other entries are ignored.

## Parameters

- **optimizer**

  *Type* → `optim.base.Optimizer | None`

  *Default* → `None`

  The sequential optimizer used for updating the weights.
- **loss**

  *Type* → `optim.losses.Loss | None`

  *Default* → `None`

  The loss function to optimize for.
- **l2**

  *Default* → `0.0`

  regularization amount used to push weights towards 0.
- **initializer**

  *Type* → `optim.initializers.Initializer | None`

  *Default* → `None`

  Weights initialization scheme.
- **clip\_gradient**

  *Default* → `1000000000000.0`

  Clips the absolute value of each gradient value.
- **seed**

  *Default* → `None`

  Random number generation seed. Set this for reproducibility.

## Attributes

- **global\_mean** ([`stats.Mean`](<https://riverml.xyz/dev/api/stats/Mean>))

  The target arithmetic mean.
- **u\_biases** (`collections.defaultdict`)

  The user bias weights.
- **i\_biases** (`collections.defaultdict`)

  The item bias weights.
- **u\_optimizer** ([`optim.base.Optimizer`](<https://riverml.xyz/dev/api/optim/base/Optimizer>))

  The sequential optimizer used for updating the user bias weights.
- **i\_optimizer** ([`optim.base.Optimizer`](<https://riverml.xyz/dev/api/optim/base/Optimizer>))

  The sequential optimizer used for updating the item bias weights.

## Examples

```python
from river import optim
from river import reco

dataset = (
    ({'user': 'Alice', 'item': 'Superman'}, 8),
    ({'user': 'Alice', 'item': 'Terminator'}, 9),
    ({'user': 'Alice', 'item': 'Star Wars'}, 8),
    ({'user': 'Alice', 'item': 'Notting Hill'}, 2),
    ({'user': 'Alice', 'item': 'Harry Potter'}, 5),
    ({'user': 'Bob', 'item': 'Superman'}, 8),
    ({'user': 'Bob', 'item': 'Terminator'}, 9),
    ({'user': 'Bob', 'item': 'Star Wars'}, 8),
    ({'user': 'Bob', 'item': 'Notting Hill'}, 2)
)

model = reco.Baseline(optimizer=optim.SGD(0.005))

for x, y in dataset:
    model.learn_one(**x, y=y)

model.predict_one(user='Bob', item='Harry Potter')
```

```text
6.538120
```

## Methods

> [!NOTE]
>
> **learn\_one**
>
> Fits a `user`-`item` pair and a real-valued target `y`.
>
> **Parameters**
>
> - **user** — `typing.Hashable`
> - **item** — `typing.Hashable`
> - **y** — `Reward`
> - **x** — `dict | None` — defaults to `None`

> [!NOTE]
>
> **predict\_one**
>
> Predicts the target value of a set of features `x`.
>
> **Parameters**
>
> - **user** — `typing.Hashable`
> - **item** — `typing.Hashable`
> - **x** — `dict | None` — defaults to `None`
>
> **Returns**
>
> `Reward`: The predicted preference from the user for the item.

> [!NOTE]
>
> **rank**
>
> Rank models by decreasing order of preference for a given user.
>
> **Parameters**
>
> - **user** — `typing.Hashable`
> - **items** — `set[typing.Hashable]`
> - **x** — `dict | None` — defaults to `None`

## References

- [Matrix factorization techniques for recommender systems](<https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf>)
