# AdjustedRand

Adjusted Rand Index.

The Adjusted Rand Index is the corrected-for-chance version of the Rand Index [^/dev/api/metrics/AdjustedRand/#1] [^/dev/api/metrics/AdjustedRand/#2]. Such a correction for chance establishes a baseline by using the expected similarity of all pair-wise comparisons between clusterings specified by a random model.

Traditionally, the Rand Index was corrected using the Permutation Model for Clustering. However, the premises of the permutation model are frequently violated; in many clustering scenarios, either the number of clusters or the size distribution of those clusters vary drastically. Variations of the adjusted Rand Index account for different models of random clusterings.

Though the Rand Index may only yield a value between 0 and 1, the Adjusted Rand index can yield negative values if the index is less than the expected index.

## Parameters

- **cm**

  *Default* → `None`

  This parameter allows sharing the same confusion matrix between multiple metrics. Sharing a confusion matrix reduces the amount of storage and computation time.

## Attributes

- **bigger\_is\_better**

  Indicate if a high value is better than a low one or not.
- **requires\_labels**

  Indicates if labels are required, rather than probabilities.
- **works\_with\_weights**

  Indicate whether the model takes into consideration the effect of sample weights

## Examples

```python
from river import metrics

y_true = [0, 0, 0, 1, 1, 1]
y_pred = [0, 0, 1, 1, 2, 2]

metric = metrics.AdjustedRand()

for yt, yp in zip(y_true, y_pred):
    metric.update(yt, yp)
    print(metric.get())
```

```text
1.0
1.0
0.0
0.0
0.09090909090909091
0.24242424242424243
```

```python
metric
```

```text
AdjustedRand: 0.242424
```

## Methods

> [!NOTE]
>
> **get**
>
> Return the current value of the metric.

> [!NOTE]
>
> **is\_better\_than**
>
> Indicate if the current metric is better than another one.
>
> **Parameters**
>
> - **other**

> [!NOTE]
>
> **revert**
>
> Revert the metric.
>
> **Parameters**
>
> - **y\_true**
> - **y\_pred**
> - **w** — defaults to `1.0`

> [!NOTE]
>
> **update**
>
> Update the metric.
>
> **Parameters**
>
> - **y\_true**
> - **y\_pred**
> - **w** — defaults to `1.0`

> [!NOTE]
>
> **works\_with**
>
> Indicates whether or not a metric can work with a given model.
>
> **Parameters**
>
> - **model** — `base.Estimator`

[^/dev/api/metrics/AdjustedRand/#1]: Wikipedia contributors. (2021, January 13). Rand index. In Wikipedia, The Free Encyclopedia, from https://en.wikipedia.org/w/index.php?title=Rand\_index&amp;oldid=1000098911

[^/dev/api/metrics/AdjustedRand/#2]: W. M. Rand (1971). "Objective criteria for the evaluation of clustering methods". Journal of the American Statistical Association. American Statistical Association. 66 (336): 846–850. arXiv:1704.01036. doi:10.2307/2284239. JSTOR 2284239.
