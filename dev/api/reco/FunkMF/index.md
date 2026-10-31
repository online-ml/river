# FunkMF

Funk Matrix Factorization for recommender systems.

The model equation is defined as:

\\\[\\hat{y}(x) = \\langle \\mathbf{v}\_u, \\mathbf{v}\_i \\rangle = \\sum\_{f=1}^{k} \\mathbf{v}\_{u, f} \\cdot \\mathbf{v}\_{i, f}\\\]

where \\(k\\) is the number of latent factors.

This model expects a dict input with a `user` and an `item` entries without any type constraint on their values (i.e. can be strings or numbers). Other entries are ignored.

## Parameters

- **n\_factors**

  *Default* → `10`

  Dimensionality of the factorization or number of latent factors.
- **optimizer** *(mutable)*

  *Type* → `optim.base.Optimizer | None`

  *Default* → `None`

  The sequential optimizer used for updating the latent factors.
- **loss** *(mutable)*

  *Type* → `optim.losses.Loss | None`

  *Default* → `None`

  The loss function to optimize for.
- **l2** *(mutable)*

  *Default* → `0.0`

  Amount of L2 regularization used to push weights towards 0.
- **initializer** *(mutable)*

  *Type* → `optim.initializers.Initializer | None`

  *Default* → `None`

  Latent factors initialization scheme.
- **clip\_gradient** *(mutable)*

  *Default* → `1000000000000.0`

  Clips the absolute value of each gradient value.
- **seed**

  *Default* → `None`

  Random number generation seed. Set this for reproducibility.

## Attributes

- **u\_latents** (`collections.defaultdict`)

  The user latent vectors randomly initialized.
- **i\_latents** (`collections.defaultdict`)

  The item latent vectors randomly initialized.
- **u\_optimizer** ([`optim.base.Optimizer`](<https://riverml.xyz/dev/api/optim/base/Optimizer>))

  The sequential optimizer used for updating the user latent weights.
- **i\_optimizer** ([`optim.base.Optimizer`](<https://riverml.xyz/dev/api/optim/base/Optimizer>))

  The sequential optimizer used for updating the item latent weights.

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

model = reco.FunkMF(
    n_factors=10,
    optimizer=optim.SGD(0.1),
    initializer=optim.initializers.Normal(mu=0., sigma=0.1, seed=11),
)

for x, y in dataset:
    model.learn_one(**x, y=y)

model.predict_one(user='Bob', item='Harry Potter')
```

```text
1.866272
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

- [Netflix update: Try this at home](<https://sifter.org/simon/journal/20061211.html>)
- [Matrix factorization techniques for recommender systems](<https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf>)
