# NormalizedMutualInfo

Normalized Mutual Information between two clusterings.

Normalized Mutual Information (NMI) is a normalized version of the Mutual Information (MI) score to scale the results between the range of 0 (no mutual information) and 1 (perfectly mutual information). In the formula, the mutual information will be normalized by a generalized mean of the entropy of true and predicted labels, defined by the `average_method`.

We note that this measure is not adjusted for chance (i.e corrected the effect of result agreement solely due to chance); as a result, the Adjusted Mutual Info Score will mostly be preferred. However, this metric is still symmetric, which means that switching true and predicted labels will not alter the score value. This fact can be useful when the metric is used to measure the agreement between two independent label solutions on the same dataset, when the ground truth remains unknown.

Another advantage of the metric is that as it is based on the calculation of entropy-related measures, it is independent of the permutation of class/cluster labels.

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

metric = metrics.NormalizedMutualInfo()
for yt, yp in zip(y_true, y_pred):
    metric.update(yt, yp)
    print(metric.get())
```

```text
1.0
1.0
0.0
0.343711
0.458065
0.515803
```

```python
metric
```

```text
NormalizedMutualInfo: 0.515804
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
