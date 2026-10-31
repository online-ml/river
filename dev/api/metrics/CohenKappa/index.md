# CohenKappa

Cohen's Kappa score.

Cohen's Kappa expresses the level of agreement between two annotators on a classification problem. It is defined as

\\\[ \\kappa = (p\_o - p\_e) / (1 - p\_e) \\\]

where \\(p\_o\\) is the empirical probability of agreement on the label assigned to any sample (prequential accuracy), and \\(p\_e\\) is the expected agreement when both annotators assign labels randomly.

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

y_true = ['cat', 'ant', 'cat', 'cat', 'ant', 'bird']
y_pred = ['ant', 'ant', 'cat', 'cat', 'ant', 'cat']

metric = metrics.CohenKappa()

for yt, yp in zip(y_true, y_pred):
    metric.update(yt, yp)

metric
```

```text
CohenKappa: 42.86%
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

- J. Cohen (1960). "A coefficient of agreement for nominal scales". Educational and Psychological Measurement 20(1):37-46. doi:10.1177/001316446002000104.
