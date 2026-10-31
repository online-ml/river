# ROCAUC

Receiver Operating Characteristic Area Under the Curve.

This metric is an approximation of the true ROC AUC. Computing the true ROC AUC would require storing all the predictions and ground truths, which isn't desirable. The approximation error is not significant as long as the predicted probabilities are well calibrated. In any case, this metric can still be used to reliably compare models between each other.

The ROC curve is discretized over thresholds spanning the `\[0, 1\]` interval, so this metric assumes `y_pred` lies in `\[0, 1\]` (e.g. a calibrated probability). If `y_pred` is on an arbitrary scale — as with most anomaly detectors, whose scores are only meaningful relative to each estimator — the thresholds miss the score range entirely and the metric degenerates (often to `0`). In that case, either normalize the scores to `\[0, 1\]` beforehand (for instance with a streaming min-max scaler, or [`preprocessing.TargetMinMaxScaler`](<https://riverml.xyz/dev/api/preprocessing/TargetMinMaxScaler>) when the scored value is a regression prediction), or use [`metrics.RollingROCAUC`](<https://riverml.xyz/dev/api/metrics/RollingROCAUC>), which ranks scores within a window and is therefore scale-invariant.

## Parameters

- **n\_thresholds**

  *Default* → `10`

  The number of thresholds used for discretizing the ROC curve. A higher value will lead to more accurate results, but will also cost more time and memory.
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

y_true = [ 0,  0,   1,  1]
y_pred = [.1, .4, .35, .8]

metric = metrics.ROCAUC()

for yt, yp in zip(y_true, y_pred):
    metric.update(yt, yp)

metric
```

```text
ROCAUC: 87.50%
```

The true ROC AUC is in fact 0.75. We can improve the accuracy by increasing the amount of thresholds. This comes at the cost more computation time and more memory usage.

```python
metric = metrics.ROCAUC(n_thresholds=20)

for yt, yp in zip(y_true, y_pred):
    metric.update(yt, yp)

metric
```

```text
ROCAUC: 75.00%
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
> - **w** — defaults to `1.0`

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
