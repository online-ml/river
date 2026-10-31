# VBeta

VBeta.

VBeta (or V-Measure) [^/dev/api/metrics/VBeta/#1] is an external entropy-based cluster evaluation measure. It provides an elegant solution to many problems that affect previously defined cluster evaluation measures including

- Dependence of clustering algorithm or dataset,
- The "problem of matching", where the clustering of only a portion of data points are evaluated, and
- Accurate evaluation and combination of two desirable aspects of clustering, homogeneity and completeness.

Based upon the calculations of homogeneity and completeness, a clustering solution's V-measure is calculated by computing the weighted harmonic mean of homogeneity and completeness,

\\\[ V\_{\\beta} = \\frac{(1 + \\beta) \\times h \\times c}{\\beta \\times h + c}. \\\]

## Parameters

- **beta**

  *Type* → `float`

  *Default* → `1.0`

  Weight of Homogeneity in the harmonic mean.
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

metric = metrics.VBeta(beta=1.0)
for yt, yp in zip(y_true, y_pred):
    metric.update(yt, yp)
    print(metric.get())
```

```text
1.0
1.0
0.0
0.3437110184854507
0.4580652856440158
0.5158037429793888
```

```python
metric
```

```text
VBeta: 51.58%
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

[^/dev/api/metrics/VBeta/#1]: Andrew Rosenberg and Julia Hirschberg (2007). V-Measure: A conditional entropy-based external cluster evaluation measure. Proceedings of the 2007 Joing Conference on Empirical Methods in Natural Language Processing and Computational Natural Language Learning, pp. 410 - 420, Prague, June 2007.
