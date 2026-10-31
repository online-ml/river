# ShrunkCovariance

Fixed-intensity shrinkage covariance, towards a constant-correlation or identity target.

Where `LedoitWolfCovariance` and `OASCovariance` estimate the shrinkage intensity from the data, `ShrunkCovariance` uses a fixed `delta` (transparent and predictable, mirroring `sklearn.covariance.ShrunkCovariance`) and offers a choice of target.

**When to use it.** When you want explicit, reproducible control over how much regularisation is applied rather than letting the data decide. The **constant-correlation** target (every pair shrunk towards the average sample correlation) is the finance-relevant default: assets share a positive baseline correlation, so pulling towards that is more sensible than pulling towards the zero-correlation identity target.

## Parameters

- **fading\_factor**

  *Type* → `float`

  *Default* → `0.5`

  Recency weight of the underlying exponentially weighted covariance.
- **delta**

  *Type* → `float`

  *Default* → `0.1`

  Shrinkage intensity in \[0, 1\] (0 = the raw covariance, 1 = the target).
- **target**

  *Type* → `str`

  *Default* → `constant_correlation`

  Either `"constant_correlation"` (default) or `"identity"`.

## Attributes

- **matrix**

## Examples

```python
from river import covariance, datasets

tickers = ["AAPL", "JPM", "XOM"]
cov = covariance.ShrunkCovariance(fading_factor=0.02, delta=0.3)
for x, _ in datasets.SP500Stocks():
    cov.update({t: x[t] for t in tickers})
cov
```

```text
       AAPL    JPM     XOM
AAPL   1.944   0.784   0.797
 JPM   0.784   1.492   0.885
 XOM   0.797   0.885   1.705
```

With `delta=0` the estimator reduces to a plain exponentially weighted covariance, and with `delta=1` it returns the target exactly.

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

- [Ledoit, O. and Wolf, M., 2003. Improved estimation of the covariance matrix of stock returns with an application to portfolio selection. Journal of Empirical Finance, 10(5), pp.603-621.](<https://www.ledoit.net/ole2.pdf>)
