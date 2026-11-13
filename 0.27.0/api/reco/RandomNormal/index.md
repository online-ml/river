# RandomNormal

Predicts random values sampled from a normal distribution.

The parameters of the normal distribution are fitted with running statistics. They parameters are independent of the user, the item, or the context, and are instead fitted globally. This recommender therefore acts as a dummy model that any serious model should easily outperform.

## Parameters

- **seed**

  *Default* → `None`

  Random number generation seed. Set this for reproducibility.

## Attributes

- **mean**

  stats.Mean
- **variance**

  stats.Var

## Examples

```python
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

model = reco.RandomNormal(seed=42)

for x, y in dataset:
    model.learn_one(**x, y=y)

model.predict_one(user='Bob', item='Harry Potter')
```

```text
6.147299621751425
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
