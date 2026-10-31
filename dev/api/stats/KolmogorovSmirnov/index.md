# KolmogorovSmirnov

Incremental Kolmogorov-Smirnov statistics.

The two-sample Kolmogorov-Smirnov test quantifies the distance between the empirical functions of two samples, with the null distribution of this statistic is calculated under the null hypothesis that the samples are drawn from the same distribution. The formula can be described as

\\\[ D\_{n, m} = \\sup\_x \\| F\_{1, n}(x) - F\_{2, m}(x) \\|. \\\]

This implementation is the incremental version of the previously mentioned statistics, with the change being in the ability to insert and remove an observation through time. This can be done using a randomized tree called Treap (or Cartesian Tree) [^/dev/api/stats/KolmogorovSmirnov/#2] with bulk operation and lazy propagation.

The implemented algorithm is able to perform the insertion and removal operations in O(logN) with high probability and calculate the Kolmogorov-Smirnov test in O(1), where N is the number of sample observations. This is a significant improvement compared to the O(N logN) cost of non-incremental implementation.

This implementation also supports the calculation of the Kuiper statistics. Different from the original Kolmogorov-Smirnov statistics, Kuiper's test [^/dev/api/stats/KolmogorovSmirnov/#3] calculates the sum of the absolute sizes of the most positive and most negative differences between the two cumulative distribution functions taken into account. As such, Kuiper's test is very sensitive in the tails as at the median.

Last but not least, this implementation is also based on the original implementation within the supplementary material of the authors of paper [^/dev/api/stats/KolmogorovSmirnov/#1], at [the following Github repository](<https://github.com/denismr/incremental-ks/tree/master>).

## Parameters

- **statistic**

  *Type* → `str`

  *Default* → `ks`

  The method used to calculate the statistic, can be either "ks" or "kuiper". The default value is set as "ks".

## Examples

```python
import numpy as np
from river import stats

stream_a = [1, 1, 2, 2, 3, 3, 4, 4]
stream_b = [1, 1, 1, 1, 2, 2, 2, 2]

incremental_ks = stats.KolmogorovSmirnov(statistic="ks")
for a, b in zip(stream_a, stream_b):
    incremental_ks.update(a, b)

incremental_ks
```

```text
KolmogorovSmirnov: 0.5
```

```python
incremental_ks.n_samples
```

```text
8
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

[^/dev/api/stats/KolmogorovSmirnov/#1]: dos Reis, D.M. et al. (2016) 'Fast unsupervised online drift detection using incremental Kolmogorov-Smirnov test', Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. doi:10.1145/2939672.2939836.

[^/dev/api/stats/KolmogorovSmirnov/#2]: C. R. Aragon and R. G. Seidel. Randomized search trees. In FOCS, pages 540–545. IEEE, 1989.

[^/dev/api/stats/KolmogorovSmirnov/#3]: Kuiper, N. H. (1960). "Tests concerning random points on a circle". Proceedings of the Koninklijke Nederlandse Akademie van Wetenschappen, Series A. 63: 38–47.
