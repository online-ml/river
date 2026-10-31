# Gaussian

Normal distribution with parameters mu and sigma.

## Parameters

- **seed**

  *Type* → `int | None`

  *Default* → `None`

  Random number generator seed for reproducibility.

## Attributes

- **mode**

  The most likely value in the distribution.
- **mu**
- **n\_samples**

  The number of observed samples.
- **sigma**

## Examples

```python
from river import proba

p = proba.Gaussian()
p.update(6)
p.update(7)

p
```

```text
𝒩(μ=6.500, σ=0.707)
```

```python
p(6.5)
```

```text
0.564189
```

```python
p.revert(7)
p
```

```text
𝒩(μ=6.000, σ=0.000)
```

## Methods

> [!NOTE]
>
> ****call****
>
> Probability mass/density function.
>
> **Parameters**
>
> - **x** — `float`

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
> **log\_pdf**
>
> Log of the probability density function.
>
> This is more efficient and numerically stable than `math.log(self(x))` because it avoids the `exp` / `sqrt` round-trip.
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
> - **w** — `float` — defaults to `1.0`

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
> - **w** — `float` — defaults to `1.0`
