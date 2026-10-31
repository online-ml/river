# Rand

Rand Index.

The Rand Index [^/dev/api/metrics/Rand/#1] [^/dev/api/metrics/Rand/#2] is a measure of the similarity between two data clusterings. Given a set of elements `S` and two partitions of `S` to compare, `X` and `Y`, define the following:

- a, the number of pairs of elements in `S` that are in the **same** subset in `X` and in the **same** subset in `Y`
- b, the number of pairs of elements in `S` that are in the **different** subset in `X` and in **different** subsets in `Y`
- c, the number of pairs of elements in `S` that are in the **same** subset in `X` and in **different** subsets in `Y`
- d, the number of pairs of elements in `S` that are in the **different** subset in `X` and in the **same** subset in `Y`

The Rand index, R, is

\\\[ R = rac{a+b}{a+b+c+d} = rac{a+b}{ rac{n(n-1)}{2}}. \\\]

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

metric = metrics.Rand()

for yt, yp in zip(y_true, y_pred):
    metric.update(yt, yp)

metric
```

```text
Rand: 0.666667
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

[^/dev/api/metrics/Rand/#1]: Wikipedia contributors. (2021, January 13). Rand index. In Wikipedia, The Free Encyclopedia, from https://en.wikipedia.org/w/index.php?title=Rand\_index&amp;oldid=1000098911

[^/dev/api/metrics/Rand/#2]: W. M. Rand (1971). "Objective criteria for the evaluation of clustering methods". Journal of the American Statistical Association. American Statistical Association. 66 (336): 846–850. arXiv:1704.01036. doi:10.2307/2284239. JSTOR 2284239.
