# RollingROCAUC

Rolling version of the Receiver Operating Characteristic Area Under the Curve.

The RollingROCAUC calculates the metric using the instances in its window of size S. It keeps a queue of the instances, when an instance is added and the queue length is equal to S, the last instance is removed. The metric has a tree with ordered instances, in order to calculate the AUC efficiently. It was implemented based on the algorithm presented in Brzezinski and Stefanowski, 2017.

The difference between this metric and the standard ROCAUC is that the latter calculates an approximation of the real metric considering all data from the beginning of the stream, while the RollingROCAUC calculates the exact value considering only the last S instances. This approach may be beneficial if it's necessary to evaluate the model's performance over time, since calculating the metric using the entire stream may hide the current performance of the classifier.

Because the AUC is computed by ranking the scores within the window, this metric is invariant to the scale of `y_pred` and needs no normalization. This makes it convenient for anomaly-detector scores, which are unbounded — unlike [`metrics.ROCAUC`](<https://riverml.xyz/dev/api/metrics/ROCAUC>), whose thresholds assume `y_pred` lies in `\[0, 1\]`.

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

metric = metrics.RollingROCAUC(window_size=4)

for yt, yp in zip(y_true, y_pred):
    metric.update(yt, yp)

metric
```

```text
RollingROCAUC: 75.00%
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
