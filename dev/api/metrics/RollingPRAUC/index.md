# RollingPRAUC

Rolling version of the Area Under the Precision-Recall Area Under Curve metric.

The RollingPRAUC calculates the AUC-PR using the instances in its window of size S. It keeps a queue of the instances, when an instance is added and the queue length is equal to S, the last instance is removed.

The AUC-PR is suitable for evaluating models under unbalanced environments. For now, the implementation can deal only with binary scenarios.

Internally, this class maintains a self-balancing binary search tree to efficiently and precisely (i.e., the result is not an approximation) compute the AUC-PR considering the current window.

This implementation is based on the paper "Efficient Prequential AUC-PR Computation" (Gomes, Grégio, Alves, and Almeida, 2023): https://doi.org/10.1109/ICMLA58977.2023.00335.

## Parameters

- **window\_size**

  *Default* → `1000`

  The max length of the window.
- **pos\_val**

  *Default* → `True`

  Value to treat as "positive".

## Attributes

- **bigger\_is\_better**

  Indicate if a high value is better than a low one or not.
- **requires\_labels**

  Indicates if labels are required, rather than probabilities.
- **works\_with\_weights**

  Indicate whether the model takes into consideration the effect of sample weights

## Examples

```python
from river import base, metrics

y_true = [ 0,  1,  0,  1,  0,  1,  0,  0,   1,  1]
y_pred = [.3, .5, .5, .7, .1, .3, .1, .4, .35, .8]

metric = metrics.RollingPRAUC(window_size=4)

for yt, yp in zip(y_true, y_pred):
    metric.update(yt, yp)

metric
```

```text
RollingPRAUC: 83.33%
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
> - **y\_true** — `bool`
> - **y\_pred** — `bool | float | dict[bool, float]`

> [!NOTE]
>
> **update**
>
> Update the metric.
>
> **Parameters**
>
> - **y\_true** — `bool`
> - **y\_pred** — `bool | float | dict[bool, float]`

> [!NOTE]
>
> **works\_with**
>
> Indicates whether or not a metric can work with a given model.
>
> **Parameters**
>
> - **model** — `base.Estimator`
