# EmpiricalPrecision

Empirical precision matrix.

The precision matrix is the inverse of the covariance matrix.

This implementation leverages the Sherman-Morrison formula. The resulting inverse covariance matrix is not guaranteed to be identical to a batch computation. However, the difference shrinks with the number of observations.

## Attributes

- **matrix**

## Examples

```python
import numpy as np
import pandas as pd
from river import covariance

np.random.seed(42)
X = pd.DataFrame(np.random.random((1000, 3)))
X.head()
```

```text
          0         1         2
0  0.374540  0.950714  0.731994
1  0.598658  0.156019  0.155995
2  0.058084  0.866176  0.601115
3  0.708073  0.020584  0.969910
4  0.832443  0.212339  0.181825
```

```python
prec = covariance.EmpiricalPrecision()
for x in X.to_dict(orient="records"):
    prec.update(x)

prec
```

```text
    0        1        2
0   12.026   -0.122   -0.214
1   -0.122   11.276   -0.026
2   -0.214   -0.026   11.632
```

```python
pd.DataFrame(np.linalg.inv(np.cov(X.T, ddof=1)))
```

```text
           0          1          2
0  12.159791  -0.124966  -0.218671
1  -0.124966  11.393394  -0.026662
2  -0.218671  -0.026662  11.756907
```

## Methods

> [!NOTE]
>
> **update**
>
> Update with a single sample.
>
> **Parameters**
>
> - **x** — `dict[typing.Any, float]`

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

## References

- [Online Estimation of the Inverse Covariance Matrix - Markus Thill](<https://markusthill.github.io/math/stats/ml/online-estimation-of-the-inverse-covariance-matrix/>)
- [Fast rank-one updates to matrix inverse? - Tim Vieira](<https://timvieira.github.io/blog/post/2021/03/25/fast-rank-one-updates-to-matrix-inverse/>)
- [Woodbury matrix identity](<https://en.wikipedia.org/wiki/Woodbury_matrix_identity>)
