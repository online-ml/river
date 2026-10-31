# Ranker

Base class for ranking models.

## Parameters

- **seed**

  *Type* → `int | None`

  *Default* → `None`

  Random number generation seed. Set this for reproducibility.

## Attributes

- **is\_contextual**

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
