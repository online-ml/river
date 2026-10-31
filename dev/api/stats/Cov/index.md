# Cov

Covariance.

## Parameters

- **ddof**

  *Type* → `int`

  *Default* → `1`

  Delta Degrees of Freedom.

## Attributes

- **n**

## Examples

```python
from river import stats

x = [-2.1,  -1,  4.3]
y = [   3, 1.1, 0.12]

cov = stats.Cov()

for xi, yi in zip(x, y):
    cov.update(xi, yi)
    print(cov.get())
```

```text
0.0
-1.044999
-4.286
```

This class has a `revert` method, and can thus be wrapped by [`utils.Rolling`](<https://riverml.xyz/dev/api/utils/Rolling>):

```python
from river import utils

x = [-2.1,  -1, 4.3, 1, -2.1,  -1, 4.3]
y = [   3, 1.1, .12, 1,    3, 1.1, .12]

rcov = utils.Rolling(stats.Cov, window_size=3)

for xi, yi in zip(x, y):
    rcov.update(xi, yi)
    print(rcov.get())
```

```text
0.0
-1.045
-4.286
-1.382
-4.589
-1.415
-4.286
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
> - **y** — `float`
> - **w** — `float` — defaults to `1.0`

> [!NOTE]
>
> **update\_many**

## Notes

The outcomes of the incremental and parallel updates are consistent with numpy's batch processing when \\(\\text{ddof} \\le 1\\).

## References

- [Wikipedia article on algorithms for calculating variance](<https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Covariance>)
- Schubert, E. and Gertz, M., 2018, July. Numerically stable parallel computation of (co-) variance. In Proceedings of the 30th International Conference on Scientific and Statistical Database Management (pp. 1-12).
