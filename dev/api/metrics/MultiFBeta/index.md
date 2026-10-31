# MultiFBeta

Multi-class F-Beta score with different betas per class.

The multiclass F-Beta score is the arithmetic average of the binary F-Beta scores of each class. The mean can be weighted by providing class weights.

## Parameters

- **betas**

  Weight of precision in the harmonic mean of each class.
- **weights**

  Class weights. If not provided then uniform weights will be used.
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

y_true = [0, 1, 2, 2, 2]
y_pred = [0, 0, 2, 2, 1]

metric = metrics.MultiFBeta(
    betas={0: 0.25, 1: 1, 2: 4},
    weights={0: 1, 1: 1, 2: 2}
)

for yt, yp in zip(y_true, y_pred):
    metric.update(yt, yp)
    print(metric)
```

```text
MultiFBeta: 100.00%
MultiFBeta: 25.76%
MultiFBeta: 62.88%
MultiFBeta: 62.88%
MultiFBeta: 46.88%
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
