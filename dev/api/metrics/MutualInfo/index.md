# MutualInfo

Mutual Information between two clusterings.

The Mutual Information [^/dev/api/metrics/MutualInfo/#1] is a measure of the similarity between two labels of the same data. Where \\(|U\_i|\\) is the number of samples in cluster \\(U\_i\\) and \\(|V\_j|\\) is the number of the samples in cluster \\(V\_j\\), the Mutual Information between clusterings \\(U\\) and \\(V\\) can be calculated as:

\\\[ MI(U,V) = \\sum\_{i=1}^{|U|} \\sum\_{v=1}^{|V|} \\frac{|U\_i \\cup V\_j|}{N} \\log \\frac{N |U\_i \\cup V\_j|}{|U\_i| |V\_j|} \\\]

This metric is independent of the absolute values of the labels: a permutation of the class or cluster label values won't change the score.

This metric is furthermore symmetric: switching `y_true` and `y_pred` will return the same score value. This can be useful to measure the agreement of two independent label assignments strategies on the same dataset when the real ground truth is not known.

The Mutual Information can be equivalently expressed as:

\\\[ MI(U,V) = H(U) - H(U | V) = H(V) - H(V | U) \\\]

where \\(H(U)\\) and \\(H(V)\\) are the marginal entropies, \\(H(U | V)\\) and \\(H(V | U)\\) are the conditional entropies.

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

metric = metrics.MutualInfo()
for yt, yp in zip(y_true, y_pred):
    metric.update(yt, yp)
    print(metric.get())
```

```text
0.0
0.0
0.0
0.215761
0.395752
0.462098
```

```python
metric
```

```text
MutualInfo: 0.462098
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

[^/dev/api/metrics/MutualInfo/#1]: Wikipedia contributors. (2021, March 17). Mutual information. In Wikipedia, The Free Encyclopedia, from https://en.wikipedia.org/w/index.php?title=Mutual\_information&amp;oldid=1012714929
