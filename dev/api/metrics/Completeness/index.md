# Completeness

Completeness Score.

Completeness [^/dev/api/metrics/Completeness/#1] is symmetrical to homogeneity. In order to satisfy the completeness criteria, a clustering must assign all of those datapoints that are members of a single class to a single cluster. To evaluate completeness, we examine the distribution cluster assignments within each class. In a perfectly complete clustering solution, each of these distributions will be completely skewed to a single cluster.

We can evaluate this degree of skew by calculating the conditional entropy of the proposed cluster distribution given the class of the component data points. However, in the worst case scenario, each class is represented by every cluster with a distribution equal to the distribution of cluster sizes. Therefore, symmetric to the calculation above, we define completeness as:

\\\[ c = \\begin{cases} 1 if H(K) = 0, \\\\ 1 - \\frac{H(K|C)}{H(K)} otherwise. \\end{cases}. \\\]

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

metric = metrics.Completeness()
for yt, yp in zip(y_true, y_pred):
    metric.update(yt, yp)
    print(metric.get())
```

```text
1.0
1.0
1.0
0.3836885465963443
0.5880325916843805
0.6666666666666667
```

```python
metric
```

```text
Completeness: 66.67%
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

[^/dev/api/metrics/Completeness/#1]: Andrew Rosenberg and Julia Hirschberg (2007). V-Measure: A conditional entropy-based external cluster evaluation measure. Proceedings of the 2007 Joing Conference on Empirical Methods in Natural Language Processing and Computational Natural Language Learning, pp. 410 - 420, Prague, June 2007.
