# BinaryDistribution

A probability distribution for discrete values.

## Parameters

- **seed**

  *Type* → `int | None`

  *Default* → `None`

  Random number generator seed for reproducibility.

## Attributes

- **mode**

  The most likely value in the distribution.
- **n\_samples**

  The number of observed samples.

## Methods

> [!NOTE]
>
> ****call****
>
> Probability mass/density function.
>
> **Parameters**
>
> - **x** — `typing.Any`

> [!NOTE]
>
> **revert**
>
> Reverts the parameters of the distribution for a given observation.
>
> **Parameters**
>
> - **x** — `bool`

> [!NOTE]
>
> **sample**
>
> Sample a random value from the distribution.

> [!NOTE]
>
> **update**
>
> Updates the parameters of the distribution given a new observation.
>
> **Parameters**
>
> - **x** — `bool`
