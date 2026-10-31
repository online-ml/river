# EWCov

Exponentially weighted covariance.

This is the bivariate counterpart of [`stats.EWVar`](<https://riverml.xyz/dev/api/stats/EWVar>). It tracks the covariance between two variables while giving more weight to recent observations, which is what you want when the relationship between the variables drifts over time (e.g. the co-movement of two asset returns during a changing market regime).

Internally it uses the identity \\(Cov(x, y) = E\\\[xy\\\] - E\\\[x\\\]E\\\[y\\\]\\), with each expectation estimated by an exponentially weighted mean ([`stats.EWMean`](<https://riverml.xyz/dev/api/stats/EWMean>)). Using the same `fading_factor` on the diagonal recovers [`stats.EWVar`](<https://riverml.xyz/dev/api/stats/EWVar>) exactly.

## Parameters

- **fading\_factor**

  *Type* → `float`

  *Default* → `0.5`

  The closer `fading_factor` is to 1 the more the statistic will adapt to recent values.

## Attributes

- **name**

## Examples

```python
from river import stats

x = [1, 3, 5, 4]
y = [2, 4, 3, 6]

ewcov = stats.EWCov(fading_factor=0.5)
for xi, yi in zip(x, y):
    ewcov.update(xi, yi)
    print(ewcov.get())
```

```text
0.0
1.0
0.5
0.625
```

## Methods

> [!NOTE]
>
> **get**
>
> Return the current value of the statistic.

> [!NOTE]
>
> **update**
>
> Update the called instance.
>
> **Parameters**
>
> - **x** — `float`
> - **y** — `float`

## References

- [Finch, T., 2009. Incremental calculation of weighted mean and variance. University of Cambridge, 4(11-5), pp.41-42.](<https://fanf2.user.srcf.net/hermes/doc/antiforgery/stats.pdf>)
- [Exponential Moving Average on Streaming Data](<https://dev.to/nestedsoftware/exponential-moving-average-on-streaming-data-4hhl>)
