# EwaCovariance

Exponentially weighted covariance matrix.

A recency-weighted estimate of the covariance: each new observation is blended into the estimate with weight `fading_factor`, so the influence of past observations decays geometrically. This is the streaming analogue of the RiskMetrics covariance and the matrix counterpart of [`stats.EWVar`](<https://riverml.xyz/dev/api/stats/EWVar>) / [`stats.EWCov`](<https://riverml.xyz/dev/api/stats/EWCov>) (the diagonal is exactly [`stats.EWVar`](<https://riverml.xyz/dev/api/stats/EWVar>) and each off-diagonal is exactly [`stats.EWCov`](<https://riverml.xyz/dev/api/stats/EWCov>)).

**When to use it.** Reach for this over `EmpiricalCovariance` when the relationships between your variables change over time. The textbook example is asset returns, whose volatilities and correlations move with the market regime: a plain empirical covariance weights a return from years ago the same as yesterday's, whereas an exponentially weighted one forgets the distant past so the risk estimate tracks current conditions. Larger `fading_factor` reacts faster (shorter memory); smaller is smoother.

## Parameters

- **fading\_factor**

  *Type* → `float`

  *Default* → `0.5`

  The closer `fading_factor` is to 1 the more weight recent observations carry. The effective memory is roughly `1 / fading_factor` observations.

## Attributes

- **matrix**

## Examples

We estimate the covariance of daily returns (in percent) for a few stocks from the [`datasets.SP500Stocks`](<https://riverml.xyz/dev/api/datasets/SP500Stocks>) dataset.

```python
from river import covariance, datasets

tickers = ["AAPL", "JPM", "XOM"]
cov = covariance.EwaCovariance(fading_factor=0.02)
for x, _ in datasets.SP500Stocks():
    cov.update({t: x[t] for t in tickers})
cov
```

```text
       AAPL    JPM     XOM
AAPL   1.944   0.766   0.760
 JPM   0.766   1.492   0.934
 XOM   0.760   0.934   1.705
```

There is also an `update_many` method to process mini-batches. It accepts any narwhals-compatible dataframe and yields the same result as feeding the rows one by one.

```python
import pandas as pd
returns = pd.DataFrame(x for x, _ in datasets.SP500Stocks())[tickers]
cov = covariance.EwaCovariance(fading_factor=0.02)
cov.update_many(returns)
cov
```

```text
       AAPL    JPM     XOM
AAPL   1.944   0.766   0.760
 JPM   0.766   1.492   0.934
 XOM   0.760   0.934   1.705
```

Individual entries are accessible by key:

```python
cov["AAPL", "JPM"]
```

```text
0.766...
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

- [RiskMetrics Technical Document (J.P. Morgan, 1996)](<https://www.msci.com/documents/10199/5915b101-4206-4ba0-aee2-3449d5c7e95a>)
