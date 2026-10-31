# EWVar

Exponentially weighted variance.

To calculate the variance we use the fact that Var(X) = Mean(x^2) - Mean(x)^2 and internally we use the exponentially weighted mean of x/x^2 to calculate this.

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

X = [1, 3, 5, 4, 6, 8, 7, 9, 11]
ewv = stats.EWVar(fading_factor=0.5)
for x in X:
    ewv.update(x)
    print(ewv.get())
```

```text
0.0
1.0
2.75
1.4375
1.984375
3.43359375
1.7958984375
2.198974609375
3.56536865234375
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

## References

- [Finch, T., 2009. Incremental calculation of weighted mean and variance. University of Cambridge, 4(11-5), pp.41-42.](<https://fanf2.user.srcf.net/hermes/doc/antiforgery/stats.pdf>)
- [Exponential Moving Average on Streaming Data](<https://dev.to/nestedsoftware/exponential-moving-average-on-streaming-data-4hhl>)
