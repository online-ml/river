# ContinuousDistribution

A probability distribution for continuous values.

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
> **cdf**
>
> Cumulative density function, i.e. P(X \<= x).
>
> **Parameters**
>
> - **x** — `float`

> [!NOTE]
>
> **revert**
>
> Reverts the parameters of the distribution for a given observation.
>
> **Parameters**
>
> - **x** — `float`

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
> - **x** — `float`
