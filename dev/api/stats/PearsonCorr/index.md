# PearsonCorr

Online Pearson correlation.

## Parameters

- **ddof**

  *Type* → `int`

  *Default* → `1`

  Delta Degrees of Freedom.

## Attributes

- **var\_x** ([`stats.Var`](<https://riverml.xyz/dev/api/stats/Var>))

  Running variance of `x`.
- **var\_y** ([`stats.Var`](<https://riverml.xyz/dev/api/stats/Var>))

  Running variance of `y`.
- **cov\_xy** ([`stats.Cov`](<https://riverml.xyz/dev/api/stats/Cov>))

  Running covariance of `x` and `y`.

## Examples

```python
from river import stats

x = [0, 0, 0, 1, 1, 1, 1]
y = [0, 1, 2, 3, 4, 5, 6]

pearson = stats.PearsonCorr()

for xi, yi in zip(x, y):
    pearson.update(xi, yi)
    print(pearson.get())
```

```text
0
0
0
0.774596
0.866025
0.878310
0.866025
```

You can also do this in a rolling fashion:

```python
from river import utils

x = [0, 0, 0, 1, 1, 1, 1]
y = [0, 1, 2, 3, 4, 5, 6]

pearson = utils.Rolling(stats.PearsonCorr, window_size=4)

for xi, yi in zip(x, y):
    pearson.update(xi, yi)
    print(pearson.get())
```

```text
0
0
0
0.7745966692414834
0.8944271909999159
0.7745966692414832
-4.712160915387242e-09
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

> [!NOTE]
>
> **update\_many**
