# Homogeneity

Homogeneity Score.

Homogeneity metric [^/dev/api/metrics/Homogeneity/#1] of a cluster labeling given a ground truth.

In order to satisfy the homogeneity criteria, a clustering must assign only those data points that are members of a single class to a single cluster. That is, the class distribution within each cluster should be skewed to a single class, that is, zero entropy. We determine how close a given clustering is to this ideal by examining the conditional entropy of the class distribution given the proposed clustering.

However, in an imperfect situation, the size of this value is dependent on the size of the dataset and the distribution of class sizes. Therefore, instead of taking the raw conditional entropy, we normalize by the maximum reduction in entropy the clustering information could provide.

As such, we define homogeneity as:

\\\[ h = \\begin{cases} 1 if H(C) = 0, \\\\ 1 - \\frac{H(C|K)}{H(C)} otherwise. \\end{cases}. \\\]

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

y_true = [1, 1, 2, 2, 3, 3]
y_pred = [1, 1, 1, 2, 2, 2]

metric = metrics.Homogeneity()
for yt, yp in zip(y_true, y_pred):
    metric.update(yt, yp)
    print(metric.get())
```

```text
1.0
1.0
0.0
0.311278
0.37515
0.42062
```

```python
metric
```

```text
Homogeneity: 42.06%
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

[^/dev/api/metrics/Homogeneity/#1]: Andrew Rosenberg and Julia Hirschberg (2007). V-Measure: A conditional entropy-based external cluster evaluation measure. Proceedings of the 2007 Joing Conference on Empirical Methods in Natural Language Processing and Computational Natural Language Learning, pp. 410 - 420, Prague, June 2007.
