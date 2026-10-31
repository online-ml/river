# Beta

Beta distribution for binary data.

A Beta distribution is very similar to a Bernoulli distribution in that it counts occurrences of boolean events. The differences lies in what is being measured. A Binomial distribution models the probability of an event occurring, whereas a Beta distribution models the probability distribution itself. In other words, it's a probability distribution over probability distributions.

## Parameters

- **alpha**

  *Type* → `int`

  *Default* → `1`

  Initial alpha parameter.
- **beta**

  *Type* → `int`

  *Default* → `1`

  Initial beta parameter.
- **seed**

  *Type* → `int | None`

  *Default* → `None`

  Random number generator seed for reproducibility.

## Attributes

- **mode**

  The most likely value in the distribution.
- **n\_samples**

  The number of observed samples.

## Examples

```python
from river import proba

successes = 81
failures = 219
beta = proba.Beta(successes, failures)

beta(.21), beta(.35)
```

```text
(0.867..., 0.165...)
```

```python
for success in range(100):
    beta.update(True)
for failure in range(200):
    beta.update(False)

beta(.21), beta(.35)
```

```text
(2.525...e-05, 0.841...)
```

`n_samples` counts the observed updates, not the prior pseudo-counts:

```python
beta.n_samples
```

```text
300
```

```python
beta.cdf(.35)
```

```text
0.994168...
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
> - **p** — `float`

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

## References

- [What is the intuition behind beta distribution?](<https://stats.stackexchange.com/questions/47771/what-is-the-intuition-behind-beta-distribution>)
