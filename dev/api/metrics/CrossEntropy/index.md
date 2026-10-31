# CrossEntropy

Multiclass generalization of the logarithmic loss.

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

y_true = [0, 1, 2, 2]
y_pred = [
    {0: 0.29450637, 1: 0.34216758, 2: 0.36332605},
    {0: 0.21290077, 1: 0.32728332, 2: 0.45981591},
    {0: 0.42860913, 1: 0.33380113, 2: 0.23758974},
    {0: 0.44941979, 1: 0.32962558, 2: 0.22095463}
]

metric = metrics.CrossEntropy()

for yt, yp in zip(y_true, y_pred):
    metric.update(yt, yp)
    print(metric.get())
```

```text
1.222454
1.169691
1.258864
1.321597
```

```python
metric
```

```text
CrossEntropy: 1.321598
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
> - **model**
