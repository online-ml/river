# BiasedMF

Biased Matrix Factorization for recommender systems.

The model equation is defined as:

\\\[\\hat{y}(x) = \\bar{y} + bu\_{u} + bi\_{i} + \\langle \\mathbf{v}\_u, \\mathbf{v}\_i \\rangle\\\]

Where \\(bu\_{u}\\) and \\(bi\_{i}\\) are respectively the user and item biases. The last term being simply the dot product between the latent vectors of the given user-item pair:

\\\[\\langle \\mathbf{v}\_u, \\mathbf{v}\_i \\rangle = \\sum\_{f=1}^{k} \\mathbf{v}\_{u, f} \\cdot \\mathbf{v}\_{i, f}\\\]

where \\(k\\) is the number of latent factors.

This model expects a dict input with a `user` and an `item` entries without any type constraint on their values (i.e. can be strings or numbers). Other entries are ignored.

## Parameters

- **n\_factors**

  *Default* → `10`

  Dimensionality of the factorization or number of latent factors.
- **bias\_optimizer** *(mutable)*

  *Type* → `optim.base.Optimizer | None`

  *Default* → `None`

  The sequential optimizer used for updating the bias weights.
- **latent\_optimizer** *(mutable)*

  *Type* → `optim.base.Optimizer | None`

  *Default* → `None`

  The sequential optimizer used for updating the latent weights.
- **loss** *(mutable)*

  *Type* → `optim.losses.Loss | None`

  *Default* → `None`

  The loss function to optimize for.
- **l2\_bias** *(mutable)*

  *Default* → `0.0`

  Amount of L2 regularization used to push bias weights towards 0.
- **l2\_latent** *(mutable)*

  *Default* → `0.0`

  Amount of L2 regularization used to push latent weights towards 0.
- **weight\_initializer** *(mutable)*

  *Type* → `optim.initializers.Initializer | None`

  *Default* → `None`

  Weights initialization scheme.
- **latent\_initializer**

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

- **global\_mean** ([`stats.Mean`](<https://riverml.xyz/dev/api/stats/Mean>))

  The target arithmetic mean.
- **u\_biases** (`collections.defaultdict`)

  The user bias weights.
- **i\_biases** (`collections.defaultdict`)

  The item bias weights.
- **u\_latents** (`collections.defaultdict`)

  The user latent vectors randomly initialized.
- **i\_latents** (`collections.defaultdict`)

  The item latent vectors randomly initialized.
- **u\_bias\_optimizer** ([`optim.base.Optimizer`](<https://riverml.xyz/dev/api/optim/base/Optimizer>))

  The sequential optimizer used for updating the user bias weights.
- **i\_bias\_optimizer** ([`optim.base.Optimizer`](<https://riverml.xyz/dev/api/optim/base/Optimizer>))

  The sequential optimizer used for updating the item bias weights.
- **u\_latent\_optimizer** ([`optim.base.Optimizer`](<https://riverml.xyz/dev/api/optim/base/Optimizer>))

  The sequential optimizer used for updating the user latent weights.
- **i\_latent\_optimizer** ([`optim.base.Optimizer`](<https://riverml.xyz/dev/api/optim/base/Optimizer>))

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

model = reco.BiasedMF(
    n_factors=10,
    bias_optimizer=optim.SGD(0.025),
    latent_optimizer=optim.SGD(0.025),
    latent_initializer=optim.initializers.Normal(mu=0., sigma=0.1, seed=71)
)

for x, y in dataset:
    model.learn_one(**x, y=y)

model.predict_one(user='Bob', item='Harry Potter')
```

```text
6.489025
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

- [Paterek, A., 2007, August. Improving regularized singular value decomposition for collaborative filtering. In Proceedings of KDD cup and workshop (Vol. 2007, pp. 5-8)](<https://www.cs.uic.edu/~liub/KDD-cup-2007/proceedings/Regular-Paterek.pdf>)
- [Matrix factorization techniques for recommender systems](<https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf>)
