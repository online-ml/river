# EwaPrecision

Exponentially weighted precision (inverse covariance) matrix.

The recency-weighted analogue of `EmpiricalPrecision`. It maintains the inverse of an exponentially weighted second-moment matrix online via a forgetting-factor Sherman-Morrison update (the recursive-least-squares trick), and applies the mean centering on read. It is genuinely online: the per-step cost is O(d^2) and the matrix is never explicitly inverted.

**When to use it.** Several methods need the *precision* matrix rather than the covariance: the Mahalanobis distance (anomaly detection), the Gaussian log-likelihood, and the weights of a Gaussian graphical model. Use this when those quantities must track a non-stationary stream, where inverting a stale covariance would lag. Like `EmpiricalPrecision`, the result is not guaranteed identical to inverting the matching covariance (there is a decaying identity prior), but the difference shrinks as observations accumulate. Requires `0 < fading_factor < 1`.

## Parameters

- **fading\_factor**

  *Type* → `float`

  *Default* → `0.5`

  Recency weight of the most recent observation. The effective memory is roughly `1 / fading_factor` observations.

## Attributes

- **matrix**

## Examples

```python
from river import covariance, datasets

tickers = ["AAPL", "JPM", "XOM"]
prec = covariance.EwaPrecision(fading_factor=0.02)
for x, _ in datasets.SP500Stocks():
    prec.update({t: x[t] for t in tickers})
prec
```

```text
       AAPL     JPM      XOM
AAPL    0.676   -0.241   -0.169
 JPM   -0.241    1.105   -0.498
 XOM   -0.169   -0.498    0.934
```

Up to the decaying prior, this approximates the inverse of the matching `EwaCovariance`:

```python
import numpy as np
cov = covariance.EwaCovariance(fading_factor=0.02)
for x, _ in datasets.SP500Stocks():
    cov.update({t: x[t] for t in tickers})
S = np.array([[cov[i, j] for j in tickers] for i in tickers])
P = np.array([[prec[i, j] for j in tickers] for i in tickers])
bool(np.allclose(P @ S, np.eye(3), atol=1e-6))
```

```text
True
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
> Any [narwhals](<https://github.com/narwhals-dev/narwhals>)-compatible eager dataframe (pandas, polars, pyarrow, ...) is accepted. The result is identical to feeding the rows one at a time with `update`, in row order.
>
> **Parameters**
>
> - **X** — `IntoDataFrame`

## References

- [Online Estimation of the Inverse Covariance Matrix - Markus Thill](<https://markusthill.github.io/math/stats/ml/online-estimation-of-the-inverse-covariance-matrix/>)
- [Recursive least squares filter](<https://en.wikipedia.org/wiki/Recursive_least_squares_filter>)
