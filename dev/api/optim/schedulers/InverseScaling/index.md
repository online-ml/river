# InverseScaling

Reduces the learning rate using a power schedule.

Assuming an initial learning rate \\(\\eta\\), the learning rate at step \\(t\\) is:

\\\[\\\\frac{eta}{(t + 1) ^ p}\\\]

where \\(p\\) is a user-defined parameter.

## Parameters

- **learning\_rate**

  *Type* → `float`
- **power**

  *Type* → `float`

  *Default* → `0.5`

## Methods

> [!NOTE]
>
> **get**
>
> Returns the learning rate at a given iteration.
>
> **Parameters**
>
> - **t** — `int`
