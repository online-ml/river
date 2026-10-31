# FowlkesMallows

Fowlkes-Mallows Index.

The Fowlkes-Mallows Index [^/dev/api/metrics/FowlkesMallows/#1] [^/dev/api/metrics/FowlkesMallows/#2] is an external evaluation method that is used to determine the similarity between two clusterings, and also a metric to measure confusion matrices. The measure of similarity could be either between two hierarchical clusterings or a clustering and a benchmark classification. A higher value for the Fowlkes-Mallows index indicates a greater similarity between the clusters and the benchmark classifications.

The Fowlkes-Mallows Index, for two cluster algorithms, is defined as:

\\\[ FM = \\sqrt{PPV \\times TPR} = \\sqrt{\\frac{TP}{TP+FP} \\times \\frac{TP}{TP+FN}} \\\]

where

- TP, FP, FN are respectively the number of true positives, false positives and false negatives;
- TPR is the True Positive Rate (or Sensitivity/Recall), PPV is the Positive Predictive Rate (or Precision).

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

metric = metrics.FowlkesMallows()

for yt, yp in zip(y_true, y_pred):
    metric.update(yt, yp)
    print(metric)
```

```text
FowlkesMallows: 0.00%
FowlkesMallows: 100.00%
FowlkesMallows: 57.74%
FowlkesMallows: 40.82%
FowlkesMallows: 35.36%
FowlkesMallows: 47.14%
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

[^/dev/api/metrics/FowlkesMallows/#1]: Wikipedia contributors. (2020, December 22). Fowlkes–Mallows index. In Wikipedia, The Free Encyclopedia, from https://en.wikipedia.org/w/index.php?title=Fowlkes%E2%80%93Mallows\_index&amp;oldid=995714222

[^/dev/api/metrics/FowlkesMallows/#2]: E. B. Fowkles and C. L. Mallows (1983). “A method for comparing two hierarchical clusterings”. Journal of the American Statistical Association
