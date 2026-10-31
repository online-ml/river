# Mean

Running mean.

## Attributes

- **n** (`float`)

  The current sum of weights. If each passed weight was 1, then this is equal to the number of seen observations.

## Examples

```python
from river import stats

X = [-5, -3, -1, 1, 3, 5]
mean = stats.Mean()
for x in X:
    mean.update(x)
    print(mean.get())
```

```text
-5.0
-4.0
-3.0
-2.0
-1.0
0.0
```

You can calculate a rolling average by wrapping a [`utils.Rolling`](<https://riverml.xyz/dev/api/utils/Rolling>) around:

```python
from river import utils

X = [1, 2, 3, 4, 5, 6]
rmean = utils.Rolling(stats.Mean, window_size=2)

for x in X:
    rmean.update(x)
    print(rmean.get())
```

```text
1.0
1.5
2.5
3.5
4.5
5.5
```

## Methods

> [!NOTE]
>
> **get**
>
> Return the current value of the statistic.

> [!NOTE]
>
> **revert**

> [!NOTE]
>
> **update**
>
> Update the called instance.
>
> **Parameters**
>
> - **x** — `float`
> - **w** — `float` — defaults to `1.0`

> [!NOTE]
>
> **update\_many**

## References

- [West, D. H. D. (1979). Updating mean and variance estimates: An improved method. Communications of the ACM, 22(9), 532-535.](<https://dl.acm.org/doi/10.1145/359146.359153>)
- [Finch, T., 2009. Incremental calculation of weighted mean and variance. University of Cambridge, 4(11-5), pp.41-42.](<https://fanf2.user.srcf.net/hermes/doc/antiforgery/stats.pdf>)
- [Chan, T.F., Golub, G.H. and LeVeque, R.J., 1983. Algorithms for computing the sample variance: Analysis and recommendations. The American Statistician, 37(3), pp.242-247.](<https://amstat.tandfonline.com/doi/abs/10.1080/00031305.1983.10483115>)
