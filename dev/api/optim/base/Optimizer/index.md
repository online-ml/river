# Optimizer

Optimizer interface.

Every optimizer inherits from this base interface.

## Parameters

- **lr** *(mutable)*

  *Type* → `int | float | Scheduler`

## Attributes

- **learning\_rate** (`float`)

  Returns the current learning rate value.

## Methods

> [!NOTE]
>
> **look\_ahead**
>
> Update a weight vector before a prediction is made.
>
> **Parameters**
>
> - **w** — `DictLike | VectorLike`
>
> **Returns**
>
> `DictLike | VectorLike`: The updated weights.

> [!NOTE]
>
> **step**
>
> Update a weight vector given a gradient.
>
> **Parameters**
>
> - **w** — `DictLike | VectorLike`
> - **g** — `DictLike | VectorLike`
>
> **Returns**
>
> `DictLike | VectorLike`: The updated weights.
