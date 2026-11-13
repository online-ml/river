# EmpiricalCovariance

Empirical covariance matrix.

## Parameters

- **ddof**

  *Type* → `int`

  *Default* → `1`

  Delta Degrees of Freedom.

## Attributes

- **matrix**

## Examples

```python
import numpy as np
import pandas as pd
from river import covariance

np.random.seed(42)
X = pd.DataFrame(np.random.random((8, 3)), columns=["red", "green", "blue"])
X
```

```text
        red     green      blue
0  0.374540  0.950714  0.731994
1  0.598658  0.156019  0.155995
2  0.058084  0.866176  0.601115
3  0.708073  0.020584  0.969910
4  0.832443  0.212339  0.181825
5  0.183405  0.304242  0.524756
6  0.431945  0.291229  0.611853
7  0.139494  0.292145  0.366362
```

```python
cov = covariance.EmpiricalCovariance()
for x in X.to_dict(orient="records"):
    cov.update(x)
cov
```

```text
        blue     green    red
 blue    0.076    0.020   -0.010
green    0.020    0.113   -0.053
  red   -0.010   -0.053    0.079
```

There is also an `update_many` method to process mini-batches. The results are identical.

```python
cov = covariance.EmpiricalCovariance()
cov.update_many(X)
cov
```

```text
        blue     green    red
 blue    0.076    0.020   -0.010
green    0.020    0.113   -0.053
  red   -0.010   -0.053    0.079
```

The covariances are stored in a dictionary, meaning any one of them can be accessed as such:

```python
cov["blue", "green"]
```

```text
Cov: 0.020292
```

Diagonal entries are variances:

```python
cov["blue", "blue"]
```

```text
Var: 0.076119
```

## Methods

> [!NOTE]
>
> **revert**
>
> Downdate with a single sample.
>
> **Parameters**
>
> - **x** — `dict[typing.Any, float]`
> - **w** — `float` — defaults to `1.0`

> [!NOTE]
>
> **update**
>
> Update with a single sample.
>
> **Parameters**
>
> - **x** — `dict[typing.Any, float]`
> - **w** — `float` — defaults to `1.0`

> [!NOTE]
>
> **update\_many**
>
> Update with a dataframe of samples.
>
> Any [narwhals](<https://github.com/narwhals-dev/narwhals>)-compatible eager dataframe (pandas, polars, pyarrow, ...) is accepted.
>
> **Parameters**
>
> - **X** — `IntoDataFrame`
