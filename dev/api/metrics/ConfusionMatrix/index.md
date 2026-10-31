# ConfusionMatrix

Confusion Matrix for binary and multi-class classification.

## Parameters

- **classes**

  *Default* → `None`

  The initial set of classes. This is optional and serves only for displaying purposes.

## Attributes

- **bigger\_is\_better**

  Indicate if a high value is better than a low one or not.
- **classes**
- **requires\_labels**

  Indicates if labels are required, rather than probabilities.
- **total\_false\_negatives**
- **total\_false\_positives**
- **total\_true\_negatives**
- **total\_true\_positives**
- **works\_with\_weights**

  Indicate whether the model takes into consideration the effect of sample weights

## Examples

```python
from river import metrics

y_true = ['cat', 'ant', 'cat', 'cat', 'ant', 'bird']
y_pred = ['ant', 'ant', 'cat', 'cat', 'ant', 'cat']

cm = metrics.ConfusionMatrix()

for yt, yp in zip(y_true, y_pred):
    cm.update(yt, yp)

cm
```

```text
       ant  bird   cat
 ant     2     0     0
bird     0     0     1
 cat     1     0     2
```

```python
cm['bird']['cat']
```

```text
1.0
```

## Methods

> [!NOTE]
>
> **false\_negatives**

> [!NOTE]
>
> **false\_positives**

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
> **support**

> [!NOTE]
>
> **true\_negatives**

> [!NOTE]
>
> **true\_positives**

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

## Notes

This confusion matrix is a 2D matrix of shape `(n_classes, n_classes)`, corresponding to a single-target (binary and multi-class) classification task.

Each row represents `true` (actual) class-labels, while each column corresponds to the `predicted` class-labels. For example, an entry in position `\[1, 2\]` means that the true class-label is 1, and the predicted class-label is 2 (incorrect prediction).

This structure is used to keep updated statistics about a single-output classifier's performance and to compute multiple evaluation metrics.
