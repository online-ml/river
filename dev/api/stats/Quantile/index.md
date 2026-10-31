# Quantile

Running quantile.

Uses the P² algorithm, which is also known as the "Piecewise-Parabolic quantile estimator". The code is inspired by LiveStat's implementation [^/dev/api/stats/Quantile/#2].

## Parameters

- **q**

  *Type* → `float`

  *Default* → `0.5`

  Determines which quantile to compute, must be comprised between 0 and 1.

## Attributes

- **name**

## Examples

```python
from river import stats
import numpy as np

np.random.seed(42 * 1337)
mu, sigma = 0, 1
s = np.random.normal(mu, sigma, 500)

median = stats.Quantile(0.5)
for x in s:
   _ = median.update(x)
print(f'The estimated value of the 50th (median) quantile is {median.get():.4f}')
```

```text
The estimated value of the 50th (median) quantile is -0.0275
```

```python
print(f'The real value of the 50th (median) quantile is {np.median(s):.4f}')
```

```text
The real value of the 50th (median) quantile is -0.0135
```

```python
percentile_17 = stats.Quantile(0.17)
for x in s:
   _ = percentile_17.update(x)
print(f'The estimated value of the 17th quantile is {percentile_17.get():.4f}')
```

```text
The estimated value of the 17th quantile is -0.8652
```

```python
print(f'The real value of the 17th quantile is {np.percentile(s,17):.4f}')
```

```text
The real value of the 17th quantile is -0.9072
```

## Methods

> [!NOTE]
>
> **get**
>
> Return the current value of the statistic.

> [!NOTE]
>
> **update**
>
> Update the called instance.
>
> **Parameters**
>
> - **x** — `float`

## References

- [The P² Algorithm for Dynamic Univariateal Computing Calculation of Quantiles and Editor Histograms Without Storing Observations](<https://www.cse.wustl.edu/~jain/papers/ftp/psqr.pdf>)
- [P² quantile estimator: estimating the median without storing values](<https://aakinshin.net/posts/p2-quantile-estimator/>)

[^/dev/api/stats/Quantile/#2]: [LiveStats](<https://github.com/cxxr/LiveStats>)
