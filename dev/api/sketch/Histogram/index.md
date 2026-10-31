# Histogram

Streaming histogram.

A histogram estimates a distribution by dividing the range in discrete bins, and counting the number of items that fall in each bin.

This version of a histogram works only with real numbers.

## Parameters

- **max\_bins**

  *Type* → `int`

  *Default* → `256`

  Maximal number of bins.

## Attributes

- **n**

  Total number of seen values.

## Examples

```python
from river import sketch
import numpy as np

np.random.seed(42)

values = np.hstack((
    np.random.normal(-3, 1, 1000),
    np.random.normal(3, 1, 1000),
))

hist = sketch.Histogram(max_bins=15)

for x in values:
    hist.update(x)

for bin in hist:
    print(bin)
```

```text
[-6.24127, -6.24127]: 1
[-5.69689, -5.19881]: 8
[-5.12390, -4.43014]: 57
[-4.42475, -3.72574]: 158
[-3.71984, -3.01642]: 262
[-3.01350, -2.50668]: 206
[-2.50329, -0.81020]: 294
[-0.80954, 0.29677]: 19
[0.40896, 0.82733]: 7
[0.84661, 1.25147]: 24
[1.26029, 2.30758]: 178
[2.31081, 3.05701]: 284
[3.05963, 3.69695]: 242
[3.69822, 5.64434]: 258
[6.13775, 6.19311]: 2
```

## Methods

> [!NOTE]
>
> **cdf**
>
> Cumulative distribution function.
>
> **Parameters**
>
> - **x** — `float`

> [!NOTE]
>
> **iter\_cdf**
>
> Yields CDF values for a sorted iterable of values.
>
> This is faster than calling `cdf` with many values.
>
> **Parameters**
>
> - **X** — `Iterable[float]`
> - **verbose** — `bool` — defaults to `False`

## References

- [Ben-Haim, Y. and Tom-Tov, E., 2010. A streaming parallel decision tree algorithm. Journal of Machine Learning Research, 11(Feb), pp.849-872.](<http://jmlr.org/papers/volume11/ben-haim10a/ben-haim10a.pdf>)
- [Go implementation](<https://github.com/VividCortex/gohistogram>)
