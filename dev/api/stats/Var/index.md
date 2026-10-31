# Var

Running variance using Welford's algorithm.

## Parameters

- **ddof**

  *Type* → `int`

  *Default* → `1`

  Delta Degrees of Freedom. The divisor used in calculations is `n - ddof`, where `n` represents the number of seen elements.

## Attributes

- **mean**

  It is necessary to calculate the mean of the data in order to calculate its variance.

## Examples

```python
from river import stats

X = [3, 5, 4, 7, 10, 12]

var = stats.Var()
for x in X:
    var.update(x)
    print(var.get())
```

```text
0.0
2.0
1.0
2.916666
7.7
12.56666
```

You can measure a rolling variance by using a [`utils.Rolling`](<https://riverml.xyz/dev/api/utils/Rolling>) wrapper:

```python
from river import utils

X = [1, 4, 2, -4, -8, 0]
rvar = utils.Rolling(stats.Var, window_size=3, ddof=1)
for x in X:
    rvar.update(x)
    print(rvar.get())
```

```text
0.0
4.5
2.333333
17.333333
25.333333
16.0
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

## Notes

The outcomes of the incremental and parallel updates are consistent with numpy's batch processing when \\(\\text{ddof} \\le 1\\).

## References

- [Wikipedia article on algorithms for calculating variance](<https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Covariance>)
- [Chan, T.F., Golub, G.H. and LeVeque, R.J., 1983. Algorithms for computing the sample variance: Analysis and recommendations. The American Statistician, 37(3), pp.242-247.](<https://amstat.tandfonline.com/doi/abs/10.1080/00031305.1983.10483115>)
- Schubert, E. and Gertz, M., 2018, July. Numerically stable parallel computation of (co-)variance. In Proceedings of the 30th International Conference on Scientific and Statistical Database Management (pp. 1-12).
