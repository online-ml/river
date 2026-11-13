# OASCovariance

Online Oracle Approximating Shrinkage (OAS) covariance.

Like `LedoitWolfCovariance`, OAS shrinks the exponentially weighted sample covariance towards a scaled identity, but uses the Chen, Wiesel, Eldar &amp; Hero (2010) shrinkage intensity, which is often better conditioned for approximately Gaussian data. The intensity is a closed form in the traces of the running covariance, applied on read; the per-step update is a plain O(d^2) exponentially weighted covariance update.

**When to use it.** The same high-dimensional / few-sample situations as `LedoitWolfCovariance`. OAS tends to shrink slightly more aggressively and is a good choice when the data are close to Gaussian; otherwise the two are interchangeable and worth comparing.

## Parameters

- **fading\_factor**

  *Type* → `float`

  *Default* → `0.5`

  Recency weight of the underlying exponentially weighted covariance. The effective sample size is roughly `1 / fading_factor`.

## Attributes

- **matrix**

## Examples

```python
import numpy as np
from river import covariance, datasets

oas = covariance.OASCovariance(fading_factor=0.05)
for x, _ in datasets.SP500Stocks():
    oas.update(x)
```

The shrinkage keeps the matrix positive-definite and invertible:

```python
names = sorted({i for i, _ in oas.matrix})
M = np.array([[oas[i, j] for j in names] for i in names])
bool(np.all(np.linalg.eigvalsh(M) > 0))
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

- [Chen, Y., Wiesel, A., Eldar, Y.C. and Hero, A.O., 2010. Shrinkage algorithms for MMSE covariance estimation. IEEE Transactions on Signal Processing, 58(10), pp.5016-5029.](<https://arxiv.org/abs/0907.4698>)
