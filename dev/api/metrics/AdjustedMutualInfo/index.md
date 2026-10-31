# AdjustedMutualInfo

Adjusted Mutual Information between two clusterings.

Adjusted Mutual Information (AMI) is an adjustment of the Mutual Information score that accounts for chance. It corrects the effect of agreement solely due to chance between clusterings, similar to the way the Adjusted Rand Index corrects the Rand Index. It is closely related to variation of information. The adjusted measure, however, is no longer metrical.

For two clusterings \\(U\\) and \\(V\\), the Adjusted Mutual Information is calculated as:

\\\[ AMI(U, V) = \\frac{MI(U, V) - E(MI(U, V))}{avg(H(U), H(V)) - E(MI(U, V))} \\\]

This metric is independent of the permutation of the class or cluster label values; furthermore, it is also symmetric. This can be useful to measure the agreement of two label assignments strategies on the same dataset, regardless of the ground truth.

However, due to the complexity of the Expected Mutual Info Score, the computation of this metric is an order of magnitude slower than most other metrics, in general.

## Parameters

- **cm**

  *Default* → `None`

  This parameter allows sharing the same confusion matrix between multiple metrics. Sharing a confusion matrix reduces the amount of storage and computation time.
- **average\_method**

  *Default* → `arithmetic`

  This parameter defines how to compute the normalizer in the denominator. Possible options include `min`, `max`, `arithmetic` and `geometric`.

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

y_true = [1, 1, 2, 2, 3, 3]
y_pred = [1, 1, 1, 2, 2, 2]

metric = metrics.AdjustedMutualInfo()
for yt, yp in zip(y_true, y_pred):
    metric.update(yt, yp)
    print(metric.get())
```

```text
1.0
1.0
0.0
0.0
0.105891
0.298792
```

```python
metric
```

```text
AdjustedMutualInfo: 0.298792
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

## References

- Wikipedia contributors. (2021, March 17). Mutual information. In Wikipedia, The Free Encyclopedia, from https://en.wikipedia.org/w/index.php?title=Mutual\_information&amp;oldid=1012714929
