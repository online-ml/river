# EWMean

Exponentially weighted mean.

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
ewm = stats.EWMean(fading_factor=0.5)
for x in X:
    ewm.update(x)
    print(ewm.get())
```

```text
1.0
2.0
3.5
3.75
4.875
6.4375
6.71875
7.859375
9.4296875
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
