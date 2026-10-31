# GeometricMean

Geometric mean score.

The geometric mean is a good indicator of a classifier's performance in the presence of class imbalance because it is independent of the distribution of examples between classes. This implementation computes the geometric mean of class-wise sensitivity (recall).

\\\[ gm = \\sqrt\\\[n\\\]{s\_1\\cdot s\_2\\cdot s\_3\\cdot \\ldots\\cdot s\_n} \\\]

where \\(s\_i\\) is the sensitivity (recall) of class \\(i\\) and \\(n\\) is the number of classes.

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

y_true = ['cat', 'ant', 'cat', 'cat', 'ant', 'bird', 'bird']
y_pred = ['ant', 'ant', 'cat', 'cat', 'ant', 'cat', 'bird']

metric = metrics.GeometricMean()

for yt, yp in zip(y_true, y_pred):
    metric.update(yt, yp)

metric
```

```text
GeometricMean: 69.34%
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

- Barandela, R. et al. “Strategies for learning in class imbalance problems”, Pattern Recognition, 36(3), (2003), pp 849-851.
