# LedoitWolfCovariance

Online Ledoit-Wolf shrinkage covariance.

Shrinks the exponentially weighted sample covariance towards a scaled identity (a sphere) by a data-driven intensity, following Ledoit &amp; Wolf (2004). The shrinkage intensity is estimated online from the dispersion of the per-observation scatter around the running covariance, and is recomputed on read; the per-step update stays O(d^2) with no stored window.

**When to use it.** The raw sample covariance is a poor estimate when the number of variables is large relative to the (effective) number of observations: it is noisy and often ill-conditioned or singular, which wrecks anything that inverts it (portfolio optimisation, Mahalanobis distances, Gaussian likelihoods). Shrinkage trades a little bias for a large reduction in variance, producing a well-conditioned, invertible matrix. Ledoit-Wolf picks the shrinkage intensity for you, so it is a strong default when there are many assets relative to clean history.

## Parameters

- **fading\_factor**

  *Type* → `float`

  *Default* → `0.5`

  Recency weight of the underlying exponentially weighted covariance. The effective sample size is roughly `1 / fading_factor`.

## Attributes

- **matrix**

## Examples

On the ten stocks of [`datasets.SP500Stocks`](<https://riverml.xyz/dev/api/datasets/SP500Stocks>) with a short effective memory, the raw covariance is poorly conditioned; Ledoit-Wolf shrinkage tames it.

```python
import numpy as np
from river import covariance, datasets

ewa = covariance.EwaCovariance(fading_factor=0.05)
lw = covariance.LedoitWolfCovariance(fading_factor=0.05)
for x, _ in datasets.SP500Stocks():
    ewa.update(x)
    lw.update(x)

def condition_number(cov):
    names = sorted({i for i, _ in cov.matrix})
    M = np.array([[cov[i, j] for j in names] for i in names])
    return np.linalg.cond(M)

condition_number(ewa) > condition_number(lw)
```

```text
np.True_
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

- [Ledoit, O. and Wolf, M., 2004. A well-conditioned estimator for large-dimensional covariance matrices. Journal of Multivariate Analysis, 88(2), pp.365-411.](<https://www.ledoit.net/honey.pdf>)
